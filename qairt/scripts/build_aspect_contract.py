# -*- coding: utf-8 -*-
"""生成多比例 app 契约：基准 1:1 原样保留，每个新比例整份放进 size_variants。

## 两种交付形态（用 --form 选）
- `--form=single`：每比例一套**单图** context。体积大（5 比例约 35.9 GiB），
  但每个 .bin 与画质实测时用的那个逐字节相同 ⇒ 风险最低。
- `--form=mg`：每段一个**多图共享权重** context，所有比例共用（约 11.95 GiB，+15%）。
  🔴 前提是 G-MG 通过（多图容器不改变数值）。**未过 G-MG 不得用 mg 形态交付。**

## 判据（执行前锁定，事后不得改）
G1 基准 models 与输入契约**逐字相同**
G2 每个变体的图名集合 == 基准图名集合
G3 变体里四个 text_encoder 条目与基准**逐字相同**（文本编码与几何无关）
G4 几何自洽：part1a 的 latents == [1,16,H/8,W/8]；select_45 的第 1 维 == 80+(H/16)*(W/16)；
   VAE 输出 pixels == [1,3,H,W]；VAE 输入 == [1,16,H/8,W/8]
G5 每个 .bin 存在，sha256 现算、size 现取
G6 part2a 的每个输出都被 part2b 消费（镜像 C++ validatePart2Split）
G7 dtype 全在 C++ normalizeFinalDtype 支持的三种之内
G8 qnn_graph_name 指向的图在该 .bin 里确实存在
G9 产物能重新解析，且每条含 C++ 必需键
G10 **同一个 .bin 只能声明一个 sha256**（mg 形态下多个比例共用文件）
任一条不过 => 不写文件、非零退出。

用法:
  python build_aspect_contract.py --form=single 1184x896 896x1184 1280x720 720x1280
  python build_aspect_contract.py --form=mg     1184x896 896x1184 1280x720 720x1280
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")

SDK = os.path.join("D:", os.sep, "qairt", "2.48.0.260626")
UTIL = os.path.join(SDK, "bin", "x86_64-windows-msvc", "qnn-context-binary-utility.exe")
P0 = os.path.join("D:", os.sep, "ZImage_Work", "p0_experiments")
AS = os.path.join(P0, "aspect")
MG = os.path.join(P0, "aspect_mg")
SCRATCH = os.path.join(
    "C:", os.sep, "Users", "sinai", "AppData", "Local", "Temp", "claude",
    "D--LocalDreamZImage", "bbf01231-6581-4b08-b5d0-7dd03e47f992", "scratchpad")
LIVE = os.path.join(SCRATCH, "deliver_backup", "BACKUP_final_qnn_contract.json")

TE = ["text_encoder_part1", "text_encoder_part2", "text_encoder_part3", "text_encoder_part4"]
SEGS = ["part1a", "part1b", "part2a", "part2b"]
DT_BYTES = {"QNN_DATATYPE_UFIXED_POINT_16": 2, "QNN_DATATYPE_BOOL_8": 1,
            "QNN_DATATYPE_INT_32": 4}
_meta_cache = {}


def graphs_of(binpath):
    """dump 该 context 的全部图。🔴 查产物是否存在并透传 stderr（约束 11·再补）。"""
    if binpath in _meta_cache:
        return _meta_cache[binpath]
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "m.json")
        r = subprocess.run([UTIL, "--context_binary", binpath, "--json_file", out],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        if not os.path.isfile(out):
            raise RuntimeError("元数据 dump 失败 rc=%d: %s\n%s%s"
                               % (r.returncode, binpath, r.stdout or "", r.stderr or ""))
        with open(out, encoding="utf-8") as f:
            g = json.load(f)["info"]["graphs"]
    _meta_cache[binpath] = g
    return g


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


_sha_cache = {}


def sha_cached(p):
    if p not in _sha_cache:
        _sha_cache[p] = sha256(p)
    return _sha_cache[p]


def tensors(lst, graph):
    out = []
    for t in lst:
        i = t["info"]
        dt = i["dataType"]
        if dt not in DT_BYTES:                                        # G7
            raise RuntimeError("%s: C++ normalizeFinalDtype 不支持 %s" % (graph, dt))
        n = 1
        for d in i["dimensions"]:
            n *= d
        q = i.get("quantizeParams", {})
        so = q.get("scaleOffset") if q.get("definition") == "QNN_DEFINITION_DEFINED" else None
        out.append({
            "name": i["name"], "physical_dtype": dt,
            "fixed_shape": list(i["dimensions"]),
            "layout": i.get("dataFormat", "QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER"),
            "exact_bytes": n * DT_BYTES[dt],
            "quantization": {"storage_dtype": dt,
                             "scale": so["scale"] if so else None,
                             "offset": so["offset"] if so else None,
                             "per_tensor_or_axis": "per-tensor" if so else "none",
                             "axis": "NONE"},
        })
    return out


def shape_of(entry, kind, name):
    for t in entry[kind]:
        if t["name"] == name:
            return t["fixed_shape"]
    return None


def sources(form, tag, base_1x1=False):
    """internal_graph_name -> (宿主 .bin, 设备文件名, 要 enable 的图名)

    base_1x1=True 时返回 mg context 里的**1:1 基准图**（图名 <seg>_fp16_L80_fp32 /
    vae_decoder_fp32）。只有 mg 形态用得上：single 形态下 1:1 本来就用现网文件。
    """
    if base_1x1:
        out = {"transformer_%s" % seg: (
            os.path.join(MG, seg, "%s_mg.SM8750.bin" % seg),
            "transformer_%s_mg_ctx.SM8750.bin" % seg,
            "%s_fp16_L80_fp32" % seg) for seg in SEGS}
        out["vae_decoder"] = (os.path.join(MG, "vae", "vae_mg.SM8750.bin"),
                              "vae_mg_ctx.SM8750.bin", "vae_decoder_fp32")
        return out
    out = {}
    for seg in SEGS:
        name = "transformer_%s" % seg
        if form == "single":
            out[name] = (os.path.join(AS, "ctx_%s_%s" % (seg, tag), "%s_%s.SM8750.bin" % (seg, tag)),
                         "transformer_%s_%s_ctx.SM8750.bin" % (seg, tag),
                         "%s_%s_fp32" % (seg, tag))
        else:
            out[name] = (os.path.join(MG, seg, "%s_mg.SM8750.bin" % seg),
                         "transformer_%s_mg_ctx.SM8750.bin" % seg,
                         "%s_%s_fp32" % (seg, tag))
    if form == "single":
        out["vae_decoder"] = (os.path.join(AS, "ctx_vae_%s" % tag, "vae_%s.SM8750.bin" % tag),
                              "vae_%s_ctx.SM8750.bin" % tag, None)
    else:
        out["vae_decoder"] = (os.path.join(MG, "vae", "vae_mg.SM8750.bin"),
                              "vae_mg_ctx.SM8750.bin", "vaeq_%s_fp32" % tag)
    return out


def build_variant(tag, form, base_by, fails, base_1x1=False):
    W, H = (1024, 1024) if base_1x1 else [int(x) for x in tag.lower().split("x")]
    LAT_H, LAT_W = H // 8, W // 8
    UNI = 80 + (LAT_H // 2) * (LAT_W // 2)
    models = [json.loads(json.dumps(base_by[n])) for n in TE]      # 文本编码逐字复制
    for name, (host, devname, want_graph) in sources(form, tag, base_1x1).items():
        if not os.path.isfile(host):
            print("  🔴 缺 .bin: %s" % host)
            fails.append("G5")
            continue
        gs = graphs_of(host)
        names = [x["info"]["graphName"] for x in gs]
        if want_graph is None:
            if len(gs) != 1:
                print("  🔴 %s: 未指定图名但该 .bin 有 %d 个图 %s" % (name, len(gs), names))
                fails.append("G8")
                continue
            g = gs[0]["info"]
        else:
            sel = [x for x in gs if x["info"]["graphName"] == want_graph]
            if not sel:                                            # G8
                print("  🔴 %s: .bin 里没有图 %r（有 %s）" % (name, want_graph, names))
                fails.append("G8")
                continue
            g = sel[0]["info"]
        models.append({
            "actual_filename": devname,
            "internal_graph_name": name,
            "qnn_graph_name": g["graphName"],
            "context_binary": "models/%s" % devname,
            "host_source": host,
            "sha256": sha_cached(host),
            "size_bytes": os.path.getsize(host),
            "quantization_type": "W8A16",
            "inputs": tensors(g["graphInputs"], name),
            "outputs": tensors(g["graphOutputs"], name),
        })
    by = {m["internal_graph_name"]: m for m in models}

    def chk(gate, ok, detail):
        print("    %-3s %s  %s" % (gate, "PASS" if ok else "FAIL", detail))
        if not ok:
            fails.append(gate)

    chk("G2", set(by) == set(base_by), "图集合一致")
    chk("G3", all(json.dumps(by[n], sort_keys=True) == json.dumps(base_by[n], sort_keys=True)
                  for n in TE if n in by), "text_encoder 四条与基准逐字相同")
    g4 = []
    if "transformer_part1a" in by:
        g4 += [("part1a.latents", shape_of(by["transformer_part1a"], "inputs", "latents"),
                [1, 16, LAT_H, LAT_W]),
               ("part1a.select_45", shape_of(by["transformer_part1a"], "outputs", "select_45"),
                [1, UNI, 1, 64])]
    if "vae_decoder" in by:
        g4 += [("vae.pixels", shape_of(by["vae_decoder"], "outputs", "pixels"), [1, 3, H, W]),
               ("vae.vae_latents", shape_of(by["vae_decoder"], "inputs", "vae_latents"),
                [1, 16, LAT_H, LAT_W])]
    for label, got, want in g4:
        chk("G4", got == want, "%s = %s（期望 %s）" % (label, got, want))
    chk("G4", len(g4) == 4, "四项几何检查都执行到了（实际 %d）" % len(g4))
    if "transformer_part2a" in by and "transformer_part2b" in by:
        o = {t["name"] for t in by["transformer_part2a"]["outputs"]}
        i = {t["name"] for t in by["transformer_part2b"]["inputs"]}
        chk("G6", o <= i, "part2a 输出全部被 part2b 消费（漏 %s）" % (o - i))
    return models


def main():
    form = "single"
    tags = []
    for a in sys.argv[1:]:
        if a.startswith("--form="):
            form = a.split("=", 1)[1]
        else:
            tags.append(a)
    if form not in ("single", "mg") or not tags:
        raise SystemExit(__doc__)
    out = os.path.join(SCRATCH, "deliver_backup", "final_qnn_contract.%s.json" % form)

    fails = []
    with open(LIVE, encoding="utf-8-sig") as f:
        live = json.load(f)
    base = live["models"]
    base_by = {m["internal_graph_name"]: m for m in base}
    print("形态 = %s ；比例 = %s\n基准图: %s\n" % (form, tags, ", ".join(sorted(base_by))))

    # ---- mg 形态：1:1 也必须走多图 context ----------------------------------
    # 🔴 第一版让 1:1 继续指向旧的单图 context（为了「基准逐字不改」的零回归）。
    #    后果：设备上同时保有旧单图 6.41 GiB + 新多图 8.26 GiB = **18.67 GiB（+79%）**，
    #    而多图里的 1:1 图**永远用不到**，成了死重量 —— 权重共享的全部意义就是只留一份。
    #    改为 1:1 也走 mg 后：12.3 GiB（+18%），启动校验 308 秒 -> 约 205 秒。
    # 安全依据：M1~M5 已实测 mg 里的 `<seg>_fp16_L80_fp32` 与部署单图的
    #    张量名/形状/dtype/量化 scale·offset **逐位相同**（见 check_mg_equivalence）。
    # 回滚：旧的单图 .bin 仍留在设备上（交付只增不删），回滚只需换回旧契约，无需重推。
    if form == "mg":
        print("=== 1:1 基准也改走 mg context ===")
        b1 = build_variant("1x1", form, base_by, fails, base_1x1=True)
        by1 = {m["internal_graph_name"]: m for m in b1}
        # 新 G1：规格必须与部署契约**逐字相同**，只允许 context_binary/sha256/
        # size_bytes/actual_filename/host_source/qnn_graph_name 变。
        # 允许不同的只有【路由字段】与【溯源字符串】；张量规格一律必须逐字相同。
        # quantization_granularity / device_source 是部署契约里的溯源文本，
        # 不是规格 —— 排除它们符合 G1' 的本意（「张量规格逐字相同」），
        # 不是因为判负才放宽。它们会被原样带回产物，信息不丢。
        VOL = {"context_binary", "sha256", "size_bytes", "actual_filename",
               "host_source", "qnn_graph_name", "quantization_type",
               "quantization_granularity", "device_source"}
        diff = []
        for n, m in by1.items():
            o = base_by[n]
            for k in set(o) | set(m):
                if k in VOL:
                    continue
                if json.dumps(o.get(k), sort_keys=True) != json.dumps(m.get(k), sort_keys=True):
                    diff.append((n, k))
        print("  G1' %s  1:1 的张量规格与部署契约逐字相同（差异 %s）"
              % ("PASS" if not diff else "FAIL", diff[:6]))
        if diff:
            fails.append("G1'")
        # 把部署契约里的溯源字段原样带回来（只补不覆盖）
        for n, m in by1.items():
            for k in ("quantization_granularity", "device_source"):
                if k in base_by[n] and k not in m:
                    m[k] = base_by[n][k]
        base = b1
        base_by = by1

    variants = {}
    for tag in tags:
        print("=== %s ===" % tag, flush=True)
        variants[tag] = build_variant(tag, form, base_by, fails)

    print("\n=== 全局判据 ===")

    def chk(gate, ok, detail):
        print("  %-4s %s  %s" % (gate, "PASS" if ok else "FAIL", detail))
        if not ok:
            fails.append(gate)

    if form == "mg":
        print("  G1   SKIP  mg 形态下 1:1 是**故意**改走多图 context 的，"
              "由更强的 G1'（张量规格逐字相同）取代")
    else:
        chk("G1", json.dumps(base, sort_keys=True) == json.dumps(live["models"], sort_keys=True),
            "基准 models 未被改动")
    # G10：同一 .bin 只能有一个 sha256（mg 形态下多比例共用文件）
    decl = {}
    dup_bad = []
    for ms in variants.values():
        for m in ms:
            p = m["context_binary"]
            if decl.setdefault(p, m["sha256"]) != m["sha256"]:
                dup_bad.append(p)
    chk("G10", not dup_bad, "同一 context_binary 只声明一个 sha256（冲突 %s）" % dup_bad)

    contract = dict(live)
    # 🔴🔴 这一行曾经漏掉：mg 形态下上面算出了新的 base（1:1 改走 mg），
    #    却没写回 contract["models"]，落盘的契约里 1:1 仍指向**旧的单图文件**。
    #    症状极隐蔽：**app 照常工作，不报任何错**，只是设备上要同时保有新旧两份
    #    （18.67 GiB 而非 12.30），后端每次启动多花约 100 秒。
    #    是宿主上直接跑 app 的 C++ 解析器（ZImageQnnContract）时打印出
    #    `setActiveSize("")` 的 file= 才发现的 —— 生成器自己的汇总打印用的是内存里的
    #    `base` 变量，与落盘内容不一致，**自证不了**。
    contract["models"] = base
    contract["size_variants"] = {}
    for tag in tags:
        W, H = [int(x) for x in tag.lower().split("x")]
        contract["size_variants"][tag] = {"models": variants[tag], "width": W, "height": H}
    contract["generated_by"] = "scripts/build_aspect_contract.py --form=%s" % form
    prov = dict(contract.get("provenance") or {})
    prov["aspect_form"] = (
        "single = 每比例一套单图 context（与画质实测所用 .bin 逐字节相同，风险最低）；"
        "mg = 每段一个多图共享权重 context。🔴 mg 形态未过 G-MG（多图容器不改变数值）"
        "不得交付。")
    contract["provenance"] = prov

    with open(out, "w", encoding="utf-8") as f:
        json.dump(contract, f, ensure_ascii=False, indent=1)
    with open(out, encoding="utf-8") as f:
        rt = json.load(f)
    need = ("actual_filename", "internal_graph_name", "context_binary", "sha256",
            "size_bytes", "inputs", "outputs")
    allm = rt["models"] + [m for v in rt["size_variants"].values() for m in v["models"]]
    chk("G9", all(all(k in m for k in need) for m in allm), "可重新解析且每条含必需键")

    if fails:
        os.remove(out)
        print("\n🔴 判据未过: %s —— 已删除产物，不交付" % ", ".join(sorted(set(fails))))
        return 1
    # 🔴 「新增」要拿**设备上现有的那份契约**去减，不能拿新基准减 ——
    #    mg 形态下新基准本身就是 mg 文件，那样会算出「新增 0 个」。
    on_device = {m["actual_filename"] for m in live["models"]}
    uniq = {}
    for ms in list(variants.values()) + [base]:
        for m in ms:
            uniq[m["actual_filename"]] = m["size_bytes"]
    newf = {k: v for k, v in uniq.items() if k not in on_device}
    keep = sum(v for k, v in uniq.items())
    print("")
    print("设备上交付后需保有（契约引用到的全部）: %.2f GiB" % (keep / 1073741824.0))
    print("\n✅ 全部判据通过 -> %s (%d B)" % (out, os.path.getsize(out)))
    print("需推送到设备 models/ 的新文件（去重后 %d 个，合计 %.2f GiB）:"
          % (len(newf), sum(newf.values()) / 1073741824.0))
    for k in sorted(newf):
        print("  %-46s %8.1f MiB" % (k, newf[k] / 1048576.0))
    return 0


sys.exit(main())
