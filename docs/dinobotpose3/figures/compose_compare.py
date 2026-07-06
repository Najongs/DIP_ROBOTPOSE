"""Stack two mesh-ladder rows into one labeled A/B comparison figure."""
from PIL import Image, ImageDraw
import os

OUT = "/home/najo/NAS/DIP/docs/dinobotpose3/figures/qualitative"
VZ = "/home/najo/NAS/DIP/3_pose_models/DINObotPose3/Eval/viz_outputs"
LBL_W = 165          # left label strip
HEAD = 34            # top header

rows = [
    ("qual_kinect_mesh_ladder_CLEANHEAD.png", "clean-trained head", "(baseline: no occlusion training)", (170, 170, 170)),
    ("qual_kinect_mesh_ladder.png",           "occ-aug head",        "(OURS: occlusion-robust)",          (255, 150, 40)),
]

imgs = [Image.open(os.path.join(VZ, f)).convert("RGB") for f, *_ in rows]
W = imgs[0].width; RH = imgs[0].height
canvas = Image.new("RGB", (LBL_W + W, HEAD + RH * 2), (18, 18, 18))
dr = ImageDraw.Draw(canvas)
dr.text((LBL_W + 8, 9), "Mesh overlay under escalating occlusion  (kinect #2400, 0 / 10 / 20 / 30 / 40 %)  —  same frame, same occluders",
        fill=(255, 255, 255))

for i, (im, (_, name, sub, col)) in enumerate(zip(imgs, rows)):
    y = HEAD + i * RH
    canvas.paste(im, (LBL_W, y))
    dr.rectangle([0, y, LBL_W - 1, y + RH - 1], fill=(28, 28, 28))
    dr.text((10, y + RH // 2 - 16), name, fill=col)
    dr.text((10, y + RH // 2 + 2), sub, fill=(150, 150, 150))
    # divider
    dr.line([0, y, LBL_W + W, y], fill=(60, 60, 60), width=1)

path = os.path.join(OUT, "qual_compare_occ_headablation.png")
canvas.save(path)
print("saved", path, canvas.size)
