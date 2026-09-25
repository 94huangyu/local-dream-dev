"""Build the publishable model zip: the working bundle minus qnn_runtime_libs/,
plus LICENSE and NOTICE. Every kept entry is streamed through unchanged and
then verified by sha256 against the source zip.
"""
import hashlib, os, shutil, zipfile

SRC = r"D:\ZImage_Work\package\ZIMAGE_sm8750_v2.zip"
HERE = os.path.dirname(os.path.abspath(__file__))
DST = os.path.join(HERE, "ZImageTurbo_A16W8_SM8750_qnn2.48.zip")
DROP = "qnn_runtime_libs/"
EXTRA = ["LICENSE", "NOTICE"]


def sha(zf, name):
    h = hashlib.sha256()
    with zf.open(name) as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


tmp = DST + ".partial"
with zipfile.ZipFile(SRC) as zin, zipfile.ZipFile(tmp, "w", allowZip64=True) as zout:
    kept = [i for i in zin.infolist() if not i.filename.startswith(DROP)]
    dropped = len(zin.infolist()) - len(kept)
    for info in kept:
        out = zipfile.ZipInfo(info.filename, info.date_time)
        out.compress_type = info.compress_type
        out.external_attr = info.external_attr
        with zin.open(info) as src, zout.open(out, "w", force_zip64=info.file_size > 0x7FFF0000) as dst:
            shutil.copyfileobj(src, dst, 1 << 22)
    for name in EXTRA:
        zout.write(os.path.join(HERE, name), name)
os.replace(tmp, DST)
print("kept %d, dropped %d (%s), added %s" % (len(kept), dropped, DROP, EXTRA))

# Verify: same entries (minus the dropped dir), byte-identical content.
with zipfile.ZipFile(SRC) as a, zipfile.ZipFile(DST) as b:
    want = {i.filename for i in a.infolist() if not i.filename.startswith(DROP)}
    got = {i.filename for i in b.infolist()} - set(EXTRA)
    assert want == got, ("entry mismatch", want ^ got)
    assert not any(n.startswith(DROP) or n.endswith(".so") for n in got)
    for n in sorted(want):
        assert sha(a, n) == sha(b, n), n
print("verified %d entries byte-identical to the working bundle; no .so inside" % len(want))
h = hashlib.sha256()
with open(DST, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
        h.update(chunk)
print("zip bytes %d  sha256 %s" % (os.path.getsize(DST), h.hexdigest()))
