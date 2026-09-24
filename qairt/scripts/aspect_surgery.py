# -*- coding: utf-8 -*-
"""#86 步 1：把 1:1 的图改成任意比例（方案 `scripts/EXP_PLAN_ASPECT.md`）。

🔴 **为什么不复用 `dit_seq_surgery.py`**：那个是**按数值重映射**（32->L、4128->4096+L），
   前提是「该数值在图里无歧义」（Tier 2 验证过 32 不与头数 30 冲突）。
   改比例**没有这个便利**：VAE 里 512 同时是**通道数**与**中间层空间边长**
   （出现在 334 个 value_info），128/256 同样。
   => 必须**按张量秩 + 位置**改，机制不同，另起脚本，不动那个已验证的工具。

规则（唯一的真相来源，别处不得再写一份）：
  · 4 维 [N, C, H, W]        => 只改第 2、3 位
  · 3 维展平 [N, C, H*W]     => 只改第 2 位，且必须能整除到已知层级
  · VAE 有多个分辨率层级：latent 边 x {1,2,4,8}（8x 上采样）
  · transformer：图像 token N=(H/2)*(W/2)，unified=N+L；patchify reshape 按位置改；
    `stack_1` 坐标表按 [1,H/2,W/2,3] 重算内容 [33,h,w]

🔴 内置**已知样本自检**（约束 8）：用 `--identity` 跑「目标尺寸 == 原尺寸」，
   产物必须与源**逐字节相同**。不同则说明脚本改了不该改的东西，立即停。

用法:
  python aspect_surgery.py --identity                 # 自检：1:1 空操作，必须逐字节相同
  python aspect_surgery.py --hw 1152x896              # 生成 4:3
"""
import hashlib
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import onnx
from onnx import numpy_helper

import canonical_sources as cs

ONNX_DIR = cs.ONNX_DIR
OLD_H = OLD_W = 1024          # 现状像素边
VAE_DOWN = 8                  # 像素 -> latent
PATCH = 2                     # latent -> token 网格
CAP = cs.DEPLOYED_L           # caption 槽数（不动）
SEGS = ["part1a", "part1b", "part2a", "part2b"]
VAE = "vae_decoder"


class Geo(object):
    """把「旧值 -> 新值」的映射集中在一处，别处不得再算一遍（#73：让副本不存在）。"""

    def __init__(self, h, w, cap):
        # 🔴 cap 必须取**源图的** caption 槽数，不是部署值。
        #    2026-08-31 实测：源是 L=32 base（unified=4128），
        #    我却拿 4096+DEPLOYED_L(80)=4176 去比 => 一个都匹配不上，
        #    tok/uni 全 0 而我差点当成「没有要改的」。
        self.cap = cap
        self.H, self.W = h, w
        self.lh, self.lw = h // VAE_DOWN, w // VAE_DOWN          # latent
        self.gh, self.gw = self.lh // PATCH, self.lw // PATCH    # token 网格
        self.olh = self.olw = OLD_H // VAE_DOWN
        self.ogh = self.ogw = self.olh // PATCH
        self.tok = self.gh * self.gw
        self.otok = self.ogh * self.ogw
        assert h % (VAE_DOWN * PATCH) == 0 and w % (VAE_DOWN * PATCH) == 0, \
            "H/W 必须能被 %d 整除" % (VAE_DOWN * PATCH)

    def vae_pair(self, h, w):
        """VAE 的 4 维空间对：旧 (olh*k, olw*k) -> 新 (lh*k, lw*k)。不匹配返回 None。"""
        if h % self.olh or w % self.olw:
            return None
        kh, kw = h // self.olh, w // self.olw
        if kh != kw or kh not in (1, 2, 4, 8):
            return None
        return (self.lh * kh, self.lw * kh)

    def vae_flat(self, n):
        """VAE 的展平维：旧 olh*olw*k^2 -> 新 lh*lw*k^2。"""
        for k in (1, 2, 4, 8):
            if n == self.olh * self.olw * k * k:
                return self.lh * self.lw * k * k
        return None

    def __repr__(self):
        return ("%dx%d  latent %dx%d  token网格 %dx%d = %d (原 %d)"
                % (self.H, self.W, self.lh, self.lw, self.gh, self.gw,
                   self.tok, self.otok))


def _read_ext(t, d):
    """读外部存储的 initializer 内容（onnx.load 用了 load_external_data=False）。"""
    # BOOL 必须在表里：`unsqueeze_3`（全 False）与 `unified_mask`（全 True）就是 bool，
    # 而它们的 dims 正含 4096 / 4128 —— 漏了会 KeyError 9。
    NP = {onnx.TensorProto.INT32: np.int32, onnx.TensorProto.INT64: np.int64,
          onnx.TensorProto.FLOAT: np.float32, onnx.TensorProto.BOOL: np.bool_,
          onnx.TensorProto.FLOAT16: np.float16}
    e = {x.key: x.value for x in t.external_data}
    with open(os.path.join(d, e["location"]), "rb") as f:
        f.seek(int(e.get("offset", 0)))
        raw = f.read(int(e["length"]))
    return np.frombuffer(raw, dtype=NP[t.data_type]).reshape([x for x in t.dims])


def detect_cap():
    """从 part1a 源图探测 caption 槽数：图输入 `caption` 的第 1 维。
    🔴 不得用 cs.DEPLOYED_L —— 那是**部署**的长度，而手术的输入是 L=32 base。"""
    m = onnx.load(cs.onnx_for("part1a"), load_external_data=False)
    for i in m.graph.input:
        if i.name == "caption":
            return i.type.tensor_type.shape.dim[1].dim_value
    raise RuntimeError("part1a 源图里找不到 caption 输入")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# 🔴 按**名字**显式处理的 IO：这些张量的空间维无歧义，用通用规则反而危险。
#    2026-08-31 事故：为躲开「头维 128」的歧义，我把 4 维空间规则整个删掉，
#    结果连 `latents` 图输入 [1,16,128,128] 也没改 => 图仍从 128x128 推出 4096 token，
#    G1 报「推断 4128 vs 现有 4064」。规则太宽会误伤，太窄会漏 —— 具名兜底最稳。
NAMED_IO = {
    "latents":     ("lat", (2, 3)),      # [1,16,H,W]  transformer 的输入/输出
    "vae_latents": ("lat", (2, 3)),      # [1,16,H,W]  VAE 输入
    "pixels":      ("pix", (2, 3)),      # [1,3,8H,8W] VAE 输出
}


def fix_named_io(vi, geo, stat):
    """按名字改 IO 的空间维。返回 True 表示已处理，调用方不再套通用规则。"""
    if vi.name not in NAMED_IO:
        return False
    kind, (a, b) = NAMED_IO[vi.name]
    d = vi.type.tensor_type.shape.dim
    if len(d) != 4:
        return False
    old = (OLD_H, OLD_W) if kind == "pix" else (geo.olh, geo.olw)
    tgt = (geo.H, geo.W) if kind == "pix" else (geo.lh, geo.lw)
    cur = (d[a].dim_value, d[b].dim_value)
    # 🔴 只有当前值**正好等于旧尺寸**才改。
    #    2026-08-31 实测：part2b 的 `latents` 输出声明是 `[16, 0, 0, 0]` —— 维度是
    #    符号/未知（dim_value=0）。无条件写入会把 0 变成 128，凭空造出一个错误的静态形状。
    #    （空操作自检抓到的：part2b `lat 2` 且字节不同。）
    if cur != old:
        return True
    if cur != tgt:
        d[a].dim_value, d[b].dim_value = tgt
        stat["vi_lat"] += 1
    return True


def fix_dims(vi, geo, is_vae, stat):
    """改 value_info / 图 IO 的形状。按秩+位置，不按数值。"""
    if fix_named_io(vi, geo, stat):
        return
    d = vi.type.tensor_type.shape.dim
    dims = [x.dim_value for x in d]
    if is_vae:
        if len(dims) == 4:
            np_ = geo.vae_pair(dims[2], dims[3])
            if np_:
                d[2].dim_value, d[3].dim_value = np_
                stat["vi4"] += 1
        # 🔴 展平维（H*W）可能出现在**任意秩的任意位置**，不只是 3 维的第 2 位。
        #    2026-08-31 实测：`view_1` 是 `[1, 16384, 1, 512]` —— 4 维、展平维在第 1 位。
        #    按值判定是安全的：`vae_flat` 只认 olh*olw*k^2 这几个精确乘积
        #    （16384/65536/262144/1048576），不会与通道数（512/256/128）混淆。
        for k in range(len(dims)):
            nf = geo.vae_flat(dims[k])
            if nf and dims[k] != nf:
                d[k].dim_value = nf
                stat["vi3"] += 1
    else:
        # 🔴🔴 **transformer 里绝不能用「4 维第 2/3 位 == 128 就是空间维」这条规则。**
        #    2026-08-31 实测：part2a 有 152 个 value_info 形如 `[1, 4128, 30, 128]`
        #    = [N, unified, 头数30, **头维128**]。30 x 128 = 3840 = hidden ✓
        #    => 那个 128 是**注意力头维**，改它等于从根上毁掉注意力。
        #    （方案 §二 已把这条歧义写成最大风险，我仍然掉了进去。）
        #    transformer 的空间信息**只体现在 token 数**（4096）与 unified（4096+CAP），
        #    以及 part1a 的 patchify reshape 与 stack_1 —— 不在 4 维张量的第 2/3 位。
        for k, v in enumerate(dims):
            if v == geo.otok:
                d[k].dim_value = geo.tok
                stat["vi_tok"] += 1
            elif v == geo.otok + geo.cap:
                d[k].dim_value = geo.tok + geo.cap
                stat["vi_uni"] += 1


def do_seg(stem, geo, is_vae, out_stem):
    src = os.path.join(ONNX_DIR, stem + ".onnx")
    m = onnx.load(src, load_external_data=False)
    g = m.graph
    stat = dict(shape=0, stack=0, ext=0, vi4=0, vi3=0, vi_tok=0, vi_uni=0,
                vi_lat=0, vi_reinfer=0)
    changed_any = (geo.lh, geo.lw) != (geo.olh, geo.olw)

    for t in list(g.initializer):
        if t.data_type not in (onnx.TensorProto.INT64, onnx.TensorProto.INT32):
            continue
        if t.data_location == 1:
            continue                          # 外部存储的大表单独处理
        a = numpy_helper.to_array(t)
        if a.size == 0 or a.size > 16:
            continue
        v = a.reshape(-1).tolist()
        nv = list(v)
        ch = False
        if is_vae:
            if len(v) == 4:                   # [N,C,H,W]
                p = geo.vae_pair(v[2], v[3])
                if p:
                    nv[2], nv[3] = p
                    ch = True
            for k in range(len(v)):           # 展平维可在任意位置（同 fix_dims 的理由）
                f = geo.vae_flat(v[k])
                if f:
                    nv[k] = f
                    ch = True
        else:
            for k, x in enumerate(v):
                if x == geo.otok:
                    nv[k] = geo.tok
                    ch = True
                elif x == geo.otok + geo.cap:
                    nv[k] = geo.tok + geo.cap
                    ch = True
            # patchify reshape [C,1,1,gh,2,gw,2]（part1a）：按**位置**改，不按数值
            if len(v) == 7 and v[4] == PATCH and v[6] == PATCH \
                    and v[3] == geo.ogh and v[5] == geo.ogw:
                nv[3], nv[5] = geo.gh, geo.gw
                ch = True
            # 🔴 unpatchify reshape [N,gh,gw,1,2,2,C]（part2b 的 `shape_120_fixed`）。
            #    2026-08-31 实测：布局与 part1a 的 patchify **不同**（网格在第 1/2 位，
            #    不是第 3/5 位），我原先只写了 part1a 那一种 => 转换器报
            #    「input 258048 != output 262144」（= 4032x64 vs 4096x64）。
            #    ⇒ 同一个语义（网格）在同一个模型里有**两种布局**，必须都覆盖。
            if len(v) == 7 and v[4] == PATCH and v[5] == PATCH \
                    and v[1] == geo.ogh and v[2] == geo.ogw:
                nv[1], nv[2] = geo.gh, geo.gw
                ch = True
            if len(v) == 3 and v[1] == geo.olh and v[2] == geo.olw:
                nv[1], nv[2] = geo.lh, geo.lw      # latents_shape
                ch = True
            # unpatchify 目标 [N,C,H,W]（part2b 的 shape_unsafe_fixed = [1,16,128,128]）。
            # 🔴 必须**两位都等于旧 latent 边**才算 —— 注意力张量是 [1,4128,30,128]，
            #    第 2 位是 30 不是 128，因此不会误命中（这条是防头维歧义的关键守卫）。
            if len(v) == 4 and v[2] == geo.olh and v[3] == geo.olw:
                nv[2], nv[3] = geo.lh, geo.lw
                ch = True
        # 🔴 值没变就**绝不重写**（2026-08-31 自检抓到）：无条件 CopyFrom 会把张量
        #    重新序列化，原本可能是 raw_data 的编码往返后变成 int64_data ⇒ 字节变了、
        #    语义没变。这会让「空操作必须逐字节相同」这道门失效，
        #    也让真实运行的 diff 里混入无关改动，事后无法归因。
        if ch and nv != v:
            t.CopyFrom(numpy_helper.from_array(
                np.array(nv, dtype=a.dtype).reshape(a.shape), name=t.name))
            stat["shape"] += 1

    # 🔴 **外部存储的常量填充张量**：上面的循环 `data_location == 1` 整个跳过了它们。
    #    2026-08-31 实测：`unsqueeze_3 [4096,1]`（全 False）、`unified_mask [1,4128]`（全 True）、
    #    `val_572 [1,32,64,2]` / `val_566 [1,32,3840]`（全 0）都在外部存储里。
    #    漏掉它们 => 图里仍有 4096/4128 => G1 报「推断 4128 vs 现有 4064」。
    #    内容是均匀常量（勘察已逐个验证），按新尺寸重建即可。
    for t in list(g.initializer):
        if t.data_location != 1 or not t.external_data:
            continue
        od = list(t.dims)
        nd = [geo.tok if x == geo.otok else
              (geo.tok + geo.cap if x == geo.otok + geo.cap else x) for x in od]
        if nd == od:
            continue
        arr = _read_ext(t, os.path.dirname(src))
        u = np.unique(arr)
        assert u.size == 1,             "%s 不是均匀常量（%d 个不同值），不能按尺寸重建，需人工处理" % (t.name, u.size)
        t.CopyFrom(numpy_helper.from_array(
            np.full(nd, u[0], dtype=arr.dtype), name=t.name))
        stat["ext"] += 1

    # 🔴 `stack_1`：烘焙的二维位置网格，**外部存储**，上面的循环跳过了它。
    #    实测内容（`inspect_aspect_consts.py`）：`stack_1[0,h,w,:] = [33, h, w]`，
    #    dims=[1,64,64,3]，通道 0 恒 33、通道 1/2 各 0..63。
    #    改比例必须按新网格重算，否则图仍按 64x64 推断（G1 实测报
    #    「推断 4128 vs 现有 4064」，正是这个缺口）。
    if not is_vae:
        for t in list(g.initializer):
            if t.name != "stack_1":
                continue
            od = list(t.dims)
            assert od == [1, geo.ogh, geo.ogw, 3], "stack_1 维度出乎意料: %s" % od
            old = _read_ext(t, os.path.dirname(src))
            grid = np.zeros((1, geo.gh, geo.gw, 3), dtype=old.dtype)
            grid[0, :, :, 0] = old[0, 0, 0, 0]              # 保留原来的模态/时间轴常量
            grid[0, :, :, 1] = np.arange(geo.gh)[:, None]   # h
            grid[0, :, :, 2] = np.arange(geo.gw)[None, :]   # w
            # 🔴 与形状常量同一条守卫：网格没变就绝不重写。
            #    否则空操作会把外部存储的张量搬成内联 => 字节变了、语义没变，
            #    「空操作必须逐字节相同」这道门就失效了（2026-08-31 第二次犯）。
            if grid.shape != old.shape or not np.array_equal(grid, old):
                t.CopyFrom(numpy_helper.from_array(grid, name="stack_1"))
                stat["stack"] += 1

    # 图 IO 必须显式改（它们是契约的一部分，推断不会去动）
    for vi in list(g.input) + list(g.output):
        fix_dims(vi, geo, is_vae, stat)

    # 🔴🔴 **value_info 整个清空、让形状推断重新生成**，不逐个打补丁。
    #    2026-08-31 走过的弯路：我先按「秩+位置」逐个改 value_info，
    #    结果被 GroupNorm 的 reshape `[0,32,-1]` 打败 —— 它声明的展平值是
    #    **C x H x W / 32**（如 512*512*512/32 = 4194304），既不是 H*W 也不是空间对，
    #    任何按模式匹配的规则都认不出来，而这类节点有十几个。
    #    value_info 是**纯派生信息**（图语义在 initializer 与节点属性里）
    #    => 清空重推是唯一不会漏的做法，也根除了 Tier 2 §34.1 那一整类事故。
    #    ⚠️ 缺失的 value_info 无害（它是可选元数据），**错误的 value_info 才致命**。
    if changed_any:
        del g.value_info[:]
        try:
            m2 = onnx.shape_inference.infer_shapes(m, strict_mode=False)
            m = m2
            stat["vi_reinfer"] = len(m.graph.value_info)
        except Exception as e:
            stat["vi_reinfer"] = -1
            print("     ⚠️ 重推失败（value_info 将为空，不影响语义）: %s"
                  % " ".join(str(e).split())[:80])

    dst = os.path.join(ONNX_DIR, out_stem + ".onnx")
    onnx.save(m, dst)
    return dst, stat


def main():
    ident = "--identity" in sys.argv
    hw = None
    for a in sys.argv[1:]:
        if a.startswith("--hw"):
            hw = a.split("=", 1)[1] if "=" in a else sys.argv[sys.argv.index(a) + 1]
    if ident:
        h = w = OLD_H
        tag = "identity"
    else:
        if not hw:
            sys.exit("用法: --identity  或  --hw 1152x896")
        w_, h_ = hw.lower().split("x")
        w, h = int(w_), int(h_)
        tag = "%dx%d" % (w, h)
    cap = detect_cap()
    geo = Geo(h, w, cap)
    print("源图 caption 槽数 = %d（从源图探测，不用 DEPLOYED_L）" % cap)
    print("目标 %s   %s" % (tag, geo))
    if ident:
        print("🔴 自检模式：产物必须与源**逐字节相同**（约束 8 的已知样本门）\n")

    ok = True
    for seg in SEGS + [VAE]:
        if seg == VAE:
            stem = VAE
            is_vae = True
        else:
            stem = os.path.splitext(os.path.basename(cs.onnx_for(seg)))[0]
            is_vae = False
        out = "%s_ar_%s" % (stem, tag)
        dst, st = do_seg(stem, geo, is_vae, out)
        line = ("  %-34s 形状 %2d｜外部 %2d｜stack %d｜IO %d｜value_info 重推 %4d"
                % (stem[:34], st["shape"], st["ext"], st["stack"],
                   st["vi_lat"], st["vi_reinfer"]))
        if ident:
            same = sha(dst) == sha(os.path.join(ONNX_DIR, stem + ".onnx"))
            print(line + ("   ✅ 逐字节相同" if same else "   🔴 **不同 => 脚本改了不该改的**"))
            ok = ok and same
            os.remove(dst)
        else:
            print(line + "  -> %s.onnx" % out)
    if ident:
        print("\n=== 自检 %s ===" % ("🟢 PASS：空操作确实是空操作" if ok else
                                    "🔴 FAIL：先修脚本，不得用它生成任何比例"))
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
