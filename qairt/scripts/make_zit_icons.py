"""Generate the 本地梦-ZIT launcher/notification icons from the user's artwork.

Adaptive icon (108dp canvas): background = solid navy sampled from the art's
edge; foreground = art scaled so the subject (radius ~650 px of 1254) sits
inside the 33dp safe circle, with alpha feathered to 0 toward the corners so
it blends into the background layer. Monochrome / notification icon = white
silhouette whose alpha comes from the green channel (the navy background has
G < ~35, the star and ring are far above that).
"""
import os
import numpy as np
from PIL import Image, ImageDraw

_HERE = os.path.dirname(os.path.abspath(__file__))           # <repo>/qairt/scripts
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
SRC = os.path.join(_REPO, "qairt", "assets", "icon_zit_source.webp")
RES = os.path.join(_REPO, "app", "src", "main", "res")
PREVIEW = os.path.join(_HERE, "icon_preview.png")               # masks / themed / notification

SUBJECT_R_PX = 655.0       # measured: G>50 max radius from centre
SAFE_R_DP = 33.5           # adaptive-icon safe zone is a 66dp circle
DENS = {"mdpi": 1, "hdpi": 1.5, "xhdpi": 2, "xxhdpi": 3, "xxxhdpi": 4}

art = Image.open(SRC).convert("RGB")
a = np.asarray(art).astype(np.float32)
N = a.shape[0]
c = (N - 1) / 2
yy, xx = np.mgrid[0:N, 0:N]
rr = np.hypot(yy - c, xx - c)

edge = np.concatenate([a[:8].reshape(-1, 3), a[-8:].reshape(-1, 3), a[:, :8].reshape(-1, 3), a[:, -8:].reshape(-1, 3)])
bg = tuple(int(v) for v in np.median(edge, 0))
bg_hex = "#FF%02X%02X%02X" % bg
print("background", bg, bg_hex)

# Foreground: art with alpha 1 inside the subject radius, fading to 0 by the corner.
fade = np.clip((N * 0.70 - rr) / (N * 0.70 - SUBJECT_R_PX), 0, 1)
# The art's straight edges are closer to the centre (627 px) than the subject's
# far reach (655 px, on a diagonal), so also fade out near the border itself.
# Subject bbox is x 93..1182 / y 98..1077 => nearest border distance ~72 px.
border = np.minimum(np.minimum(yy, N - 1 - yy), np.minimum(xx, N - 1 - xx))
fade = fade * np.clip(border / 70.0, 0, 1)
fg_full = np.dstack([a, fade * 255]).astype(np.uint8)
fg_img = Image.fromarray(fg_full, "RGBA")

# Silhouette alpha from green channel.
sil_alpha = np.clip((a[..., 1] - 45) / (170 - 45), 0, 1) * (rr <= SUBJECT_R_PX + 10)
sil = np.dstack([np.full_like(sil_alpha, 255)] * 3 + [sil_alpha * 255]).astype(np.uint8)
sil_img = Image.fromarray(sil, "RGBA")


def on_canvas(img, canvas_px):
    """Place the square art so its subject radius maps to SAFE_R_DP on a 108dp canvas."""
    scale = (SAFE_R_DP / 108.0) * canvas_px / SUBJECT_R_PX
    size = int(round(N * scale))
    small = img.resize((size, size), Image.LANCZOS)
    out = Image.new("RGBA", (canvas_px, canvas_px), (0, 0, 0, 0))
    off = (canvas_px - size) // 2
    out.alpha_composite(small, (off, off))
    return out


for name, k in DENS.items():
    d = os.path.join(RES, "mipmap-" + name)
    os.makedirs(d, exist_ok=True)
    px = int(round(108 * k))
    on_canvas(fg_img, px).save(os.path.join(d, "ic_launcher_zit_foreground.png"), optimize=True)
    on_canvas(sil_img, px).save(os.path.join(d, "ic_launcher_zit_monochrome.png"), optimize=True)
    # Notification small icon: 24dp, subject fills ~22dp.
    sd = os.path.join(RES, "drawable-" + name)
    os.makedirs(sd, exist_ok=True)
    npx = int(round(24 * k))
    crop_r = SUBJECT_R_PX + 5
    box = (int(c - crop_r), int(c - crop_r), int(c + crop_r), int(c + crop_r))
    s = sil_img.crop(box).resize((int(round(22 * k)),) * 2, Image.LANCZOS)
    note = Image.new("RGBA", (npx, npx), (0, 0, 0, 0))
    note.alpha_composite(s, ((npx - s.width) // 2, (npx - s.height) // 2))
    note.save(os.path.join(sd, "ic_stat_zit.png"), optimize=True)

# Preview: circle / squircle / rounded-square masks, plus themed + notification.
P = 432
bgl = Image.new("RGBA", (P, P), bg + (255,))
comp = bgl.copy()
comp.alpha_composite(on_canvas(fg_img, P))
vis = comp.crop((P // 6, P // 6, P - P // 6, P - P // 6))  # visible 72dp
V = vis.width
tiles = []
for shape in ("circle", "rounded", "square"):
    m = Image.new("L", (V, V), 0)
    dr = ImageDraw.Draw(m)
    if shape == "circle":
        dr.ellipse((0, 0, V - 1, V - 1), fill=255)
    elif shape == "rounded":
        dr.rounded_rectangle((0, 0, V - 1, V - 1), radius=V // 4, fill=255)
    else:
        dr.rectangle((0, 0, V - 1, V - 1), fill=255)
    t = Image.new("RGBA", (V, V), (235, 235, 235, 255))
    t.paste(vis, (0, 0), m)
    tiles.append(t)
themed = Image.new("RGBA", (P, P), (40, 60, 40, 255))
mono = on_canvas(sil_img, P)
tint = Image.new("RGBA", (P, P), (200, 230, 190, 255))
themed.paste(tint, (0, 0), mono.split()[3])
themed = themed.crop((P // 6, P // 6, P - P // 6, P - P // 6))
m = Image.new("L", (V, V), 0)
ImageDraw.Draw(m).ellipse((0, 0, V - 1, V - 1), fill=255)
t = Image.new("RGBA", (V, V), (235, 235, 235, 255))
t.paste(themed, (0, 0), m)
tiles.append(t)
note = Image.open(os.path.join(RES, "drawable-xxxhdpi", "ic_stat_zit.png"))
nt = Image.new("RGBA", (V, V), (30, 30, 30, 255))
big = note.resize((V // 2, V // 2), Image.NEAREST)
nt.alpha_composite(big, (V // 4, V // 4))
tiles.append(nt)
sheet = Image.new("RGBA", (V * len(tiles) + 20 * (len(tiles) + 1), V + 40), (235, 235, 235, 255))
for i, t in enumerate(tiles):
    sheet.alpha_composite(t, (20 + i * (V + 20), 20))
sheet.save(PREVIEW)
print("ok", PREVIEW)
