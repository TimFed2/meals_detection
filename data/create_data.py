import os
import random
import itertools
import math
import shutil
import cv2
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm

# === Пути ===
PROJECT_ROOT   = Path(__file__).resolve().parent.parent  # meal_detection/
DATA_DIR       = PROJECT_ROOT / "data"
TRAIN_DATA_DIR = DATA_DIR / "train_data"
TRAIN_IMG_DIR  = TRAIN_DATA_DIR / "train" / "images"
TRAIN_LBL_DIR  = TRAIN_DATA_DIR / "train" / "labels"
VAL_IMG_DIR    = TRAIN_DATA_DIR / "val"   / "images"
VAL_LBL_DIR    = TRAIN_DATA_DIR / "val"   / "labels"
QA_DIR         = DATA_DIR / "train_data_examples"

# Создать папки
for d in [TRAIN_IMG_DIR, TRAIN_LBL_DIR, VAL_IMG_DIR, VAL_LBL_DIR, QA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === Параметры ===
TRAIN_RATIO    = 0.85
TARGET_PER_K   = 430
ROT_RANGE_DISH = 90
ROT_RANGE_OBJ  = 10
SCALE_RANGE    = (0.9, 1.1)
HSV_RANGE      = (0.9, 1.1)
OCC_FRAC       = (0.1, 0.6)
OBJ_PROB       = 0.15
MAX_GEN_TRIES  = 10
LINES = [ (0,2475), (320,750), (1880,750), (2160,2075) ]

CLASS_NAMES = [
    "tea", "greek_salad", "shrimp_caesar_salad",
    "lamb_meat", "chicken_steak", "borsch", "yellow_soup"
]
CLASS_ID    = {n:i for i,n in enumerate(CLASS_NAMES)}
ITER_DICT   = {k: max(1, round(TARGET_PER_K / math.comb(7, k))) for k in range(1,8)}

# === Функции ===
def is_over(x, y):
    # Точка должна быть «слева» от всех направленных отрезков
    for (ax,ay), (bx,by) in zip(LINES, LINES[1:]):
        if (bx-ax)*(y-ay) - (by-ay)*(x-ax) <= 0:
            return False
    return True

def random_pt(W, H):
    while True:
        x, y = random.uniform(0, W), random.uniform(0, H)
        if is_over(x, y):
            return int(x), int(y)

def augment(img, max_rot):
    h, w = img.shape[:2]
    angle = random.uniform(-max_rot, max_rot)
    scale = random.uniform(*SCALE_RANGE)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    cos, sin = abs(M[0,0]), abs(M[0,1])
    nw = int(h*sin + w*cos)
    nh = int(h*cos + w*sin)
    M[0,2] += nw/2 - w/2
    M[1,2] += nh/2 - h/2
    out = cv2.warpAffine(
        img, M, (nw, nh), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0)
    )
    bgr = out[...,:3]
    alpha = out[...,3]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[...,1] *= random.uniform(*HSV_RANGE)
    hsv[...,2] *= random.uniform(*HSV_RANGE)
    b2 = cv2.cvtColor(np.clip(hsv,0,255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return np.dstack([b2, alpha])

def tight_bbox(canvas, patch, x0, y0):
    H, W = canvas.shape[:2]
    h, w = patch.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[y0:y0+h, x0:x0+w] = (patch[...,3] > 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    pts = np.vstack(cnts)
    x, y, w2, h2 = cv2.boundingRect(pts)
    return x, y, x + w2, y + h2

# === Загрузка ресурсов ===
# Фон
bg = cv2.imread(str(DATA_DIR / "table.jpg"))
Hf, Wf = bg.shape[:2]
# Блюда
meal_imgs = {}
for fn in (DATA_DIR / "meals_crop").glob("*.png"):
    name = fn.name.replace('_empty','').rsplit('.',1)[0]
    meal_imgs.setdefault(name, []).append(cv2.imread(str(fn), cv2.IMREAD_UNCHANGED))
# Объекты
objects = {}
for nm in ["phone","cup"]:
    p = DATA_DIR / "objects" / f"{nm}.png"
    if p.exists():
        objects[nm] = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)

# === Генерация ===
metadata = []
total = sum(ITER_DICT.values())
with tqdm(total=total, desc="Generating") as pbar:
    for k in range(1,8):
        for combo in itertools.combinations(CLASS_NAMES, k):
            for idx in range(ITER_DICT[k]):
                for _ in range(MAX_GEN_TRIES):
                    img = bg.copy()
                    annotations = []
                    boxes = []
                    ok = True
                    # блюда
                    for dish in combo:
                        patch = random.choice(meal_imgs[dish])
                        patch = augment(patch, ROT_RANGE_DISH)
                        pre_occ = patch.copy()  # bbox без occlusion
                        h, w = patch.shape[:2]
                        # occlusion-квадрат
                        side = int(random.uniform(*OCC_FRAC) * min(h, w))
                        if side > 0:
                            dx = random.randint(0, w-side)
                            dy = random.randint(0, h-side)
                            patch[dy:dy+side, dx:dx+side, 3] = 0
                        x, y = random_pt(Wf, Hf)
                        x0, y0 = x - w//2, y - h//2
                        x1, y1 = x0 + w, y0 + h
                        if x0<0 or y0<0 or x1>Wf or y1>Hf:
                            ok = False
                            break
                        # проверка по линиям и пересечений
                        for vx, vy in [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]:
                            if not is_over(vx, vy): ok=False; break
                        for bx0,by0,bx1,by1 in boxes:
                            if not (x1<=bx0 or x0>=bx1 or y1<=by0 or y0>=by1): ok=False; break
                        if not ok: break
                        # смешивание
                        alpha = patch[...,3]/255.0
                        for c in range(3):
                            img[y0:y1, x0:x1, c] = alpha * patch[...,c] + (1-alpha) * img[y0:y1, x0:x1, c]
                        tb = tight_bbox(img, pre_occ, x0, y0)
                        if tb is None:
                            ok = False
                            break
                        boxes.append(tb)
                        xc = ((tb[0]+tb[2])/(2*Wf))
                        yc = ((tb[1]+tb[3])/(2*Hf))
                        bw = (tb[2]-tb[0])/Wf
                        bh = (tb[3]-tb[1])/Hf
                        annotations.append(f"{CLASS_ID[dish]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                    if not ok:
                        continue
                    # объекты без аннотаций
                    for nm, obj in objects.items():
                        if random.random() > OBJ_PROB:
                            continue
                        pch = augment(obj, ROT_RANGE_OBJ)
                        h2, w2 = pch.shape[:2]
                        x, y = random_pt(Wf, Hf)
                        x0, y0 = x - w2//2, y - h2//2
                        x1, y1 = x0 + w2, y0 + h2
                        if 0<=x0< Wf-w2 and 0<=y0< Hf-h2:
                            alpha = pch[...,3]/255.0
                            for c in range(3):
                                img[y0:y1, x0:x1, c] = alpha*pch[...,c] + (1-alpha)*img[y0:y1, x0:x1, c]
                    # сохранить
                    folder_img = TRAIN_IMG_DIR if random.random() < TRAIN_RATIO else VAL_IMG_DIR
                    folder_lbl = TRAIN_LBL_DIR if folder_img is TRAIN_IMG_DIR else VAL_LBL_DIR
                    fname = f"{k}_{'-'.join(combo)}_{idx}.jpg"
                    cv2.imwrite(str(folder_img / fname), img)
                    with open(folder_lbl / fname.replace('.jpg','.txt'), 'w') as f:
                        f.write("\n".join(annotations))
                    metadata.append((folder_img/fname, annotations))
                    pbar.update(1)
                    break

# === data.yaml ===
train_rel = os.path.relpath(TRAIN_IMG_DIR, PROJECT_ROOT)
val_rel   = os.path.relpath(VAL_IMG_DIR,   PROJECT_ROOT)
cfg = {
    'train': f"./{train_rel}",
    'val':   f"./{val_rel}",
    'nc':    len(CLASS_NAMES),
    'names': {i:n for i,n in enumerate(CLASS_NAMES)}
}
with open(TRAIN_DATA_DIR / 'data.yaml', 'w') as f:
    yaml.dump(cfg, f, sort_keys=False)

# === QA примеры ===
qa_samples = random.sample(metadata, min(15, len(metadata)))
for img_path, ann in qa_samples:
    img = cv2.imread(str(img_path)); H,W = img.shape[:2]
    for line in ann:
        cid, xc, yc, bw, bh = map(float, line.split())
        x0 = int((xc - bw/2)*W); y0 = int((yc - bh/2)*H)
        x1 = int((xc + bw/2)*W); y1 = int((yc + bh/2)*H)
        cv2.rectangle(img, (x0,y0), (x1,y1), (0,255,0), 2)
    cv2.imwrite(str(QA_DIR / img_path.name), img)
