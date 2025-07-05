#!/usr/bin/env python3
"""
Генерация синтетического датасета с прогресс-баром:
- Сохранение сразу в data/train_data/{train,val} (85/15)
- QA-примеры в data/train_data_examples
- Отрисовка прогресса с tqdm
"""
import os, random, itertools, math, cv2, numpy as np, shutil, yaml
from pathlib import Path
from tqdm import tqdm

# Пути
BASE_DIR      = Path(__file__).resolve().parent
DATA_DIR      = BASE_DIR
MEALS_DIR     = DATA_DIR / "meals_crop"
TABLE_PATH    = DATA_DIR / "table.jpg"
OBJECTS_DIR   = DATA_DIR / "objects"
TRAIN_DATA    = DATA_DIR / "train_data"
TRAIN_IMG_DIR = TRAIN_DATA / "train" / "images"
TRAIN_LBL_DIR = TRAIN_DATA / "train" / "labels"
VAL_IMG_DIR   = TRAIN_DATA / "val"   / "images"
VAL_LBL_DIR   = TRAIN_DATA / "val"   / "labels"
QA_DIR        = DATA_DIR  / "train_data_examples"
for d in [TRAIN_IMG_DIR,TRAIN_LBL_DIR,VAL_IMG_DIR,VAL_LBL_DIR,QA_DIR]: d.mkdir(parents=True, exist_ok=True)

# Параметры
TRAIN_RATIO    = 0.85
ROT_DISH       = 90
ROT_OBJ        = 10
SCALE_RANGE    = (0.9,1.1)
HSV_RANGE      = (0.9,1.1)
OCC_FRAC_RANGE = (0.1,0.6)
OBJ_PROB       = 0.15
MAX_GEN        = 10
TARGET_PER_K   = 430
LINES          = [(0,2475),(320,750),(1880,750),(2160,2075)]
CLASS_NAMES    = ["tea","greek_salad","shrimp_caesar_salad","lamb_meat","chicken_steak","borsch","yellow_soup"]
CLASS_IDS      = {n:i for i,n in enumerate(CLASS_NAMES)}
ITER_COUNTS    = {k:max(1,round(TARGET_PER_K/math.comb(7,k))) for k in range(1,8)}

# Функции геометрии и аугментаций
def is_over(x,y):
    for (ax,ay),(bx,by) in zip(LINES,LINES[1:]):
        if (bx-ax)*(y-ay)-(by-ay)*(x-ax)<=0: return False
    return True

def rand_pt(W,H):
    while True:
        x,y=random.uniform(0,W),random.uniform(0,H)
        if is_over(x,y): return int(x),int(y)

def augment(img,rot_deg):
    h,w=img.shape[:2]
    ang=random.uniform(-rot_deg,rot_deg); sc=random.uniform(*SCALE_RANGE)
    M=cv2.getRotationMatrix2D((w/2,h/2),ang,sc)
    cos,sin=abs(M[0,0]),abs(M[0,1]); nw=int(h*sin+w*cos); nh=int(h*cos+w*sin)
    M[0,2]+=nw/2-w/2; M[1,2]+=nh/2-h/2
    out=cv2.warpAffine(img,M,(nw,nh),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))
    b,a=out[...,:3],out[...,3]
    hsv=cv2.cvtColor(b,cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[...,1]*=random.uniform(*HSV_RANGE); hsv[...,2]*=random.uniform(*HSV_RANGE)
    b2=cv2.cvtColor(np.clip(hsv,0,255).astype(np.uint8),cv2.COLOR_HSV2BGR)
    return np.dstack([b2,a])

def tight_bbox(canvas,patch,x0,y0):
    H,W=canvas.shape[:2]; h,w=patch.shape[:2]
    mask=np.zeros((H,W),np.uint8)
    mask[y0:y0+h,x0:x0+w]=(patch[...,3]>0).astype(np.uint8)*255
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    pts=np.vstack(cnts); x,y,w2,h2=cv2.boundingRect(pts)
    return x,y,x+w2,y+h2

# Загрузка ресурсов
meals={}
for p in MEALS_DIR.glob("*.png"):
    name=p.name.replace('_empty','').rsplit('.',1)[0]
    meals.setdefault(name,[]).append(cv2.imread(str(p),cv2.IMREAD_UNCHANGED))
objects={}
for nm in ["phone","cup"]:
    f=OBJECTS_DIR/f"{nm}.png"
    if f.exists(): objects[nm]=cv2.imread(str(f),cv2.IMREAD_UNCHANGED)

# Фон
bg=cv2.imread(str(TABLE_PATH)); Hf,Wf=bg.shape[:2]

# Подсчет общего числа для прогресс-бара
total = sum(ITER_COUNTS[k]*math.comb(7,k) for k in range(1,8))

# Генерация
metadata=[]
pbar = tqdm(total=total, desc="Generating")
for k in range(1,8):
    for combo in itertools.combinations(CLASS_NAMES,k):
        for idx in range(ITER_COUNTS[k]):
            for _ in range(MAX_GEN):
                img=bg.copy(); ann=[]; boxes=[]; ok=True
                # блюда
                for dish in combo:
                    patch=random.choice(meals[dish]); patch=augment(patch,ROT_DISH)
                    h,w=patch.shape[:2]
                    # occlusion
                    side=int(random.uniform(*OCC_FRAC_RANGE)*min(h,w))
                    if side>0:
                        dx,dy=random.randint(0,w-side),random.randint(0,h-side)
                        patch[dy:dy+side,dx:dx+side,3]=0
                    x,y=rand_pt(Wf,Hf)
                    x0,y0=x-w//2,y-h//2; x1,y1=x0+w,y0+h
                    if x0<0 or y0<0 or x1>Wf or y1>Hf: ok=False; break
                    for vx,vy in [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]:
                        if not is_over(vx,vy): ok=False; break
                    for bx0,by0,bx1,by1 in boxes:
                        if not(x1<=bx0 or x0>=bx1 or y1<=by0 or y0>=by1): ok=False; break
                    if not ok: break
                    alpha=patch[...,3]/255.0
                    for c in range(3): img[y0:y1,x0:x1,c]=alpha*patch[...,c]+(1-alpha)*img[y0:y1,x0:x1,c]
                    tb=tight_bbox(img,patch,x0,y0)
                    if tb is None: ok=False; break
                    boxes.append(tb)
                    cx,cy=((tb[0]+tb[2])/(2*Wf)),((tb[1]+tb[3])/(2*Hf))
                    bw,bh=(tb[2]-tb[0])/Wf,(tb[3]-tb[1])/Hf
                    ann.append(f"{CLASS_IDS[dish]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                if not ok: continue
                # объекты
                for nm,obj in objects.items():
                    if random.random()>OBJ_PROB: continue
                    pch=augment(obj,ROT_OBJ); h2,w2=pch.shape[:2]
                    x,y=rand_pt(Wf,Hf); x0,y0=x-w2//2,y-h2//2; x1,y1=x0+w2,y0+h2
                    if 0<=x0 and 0<=y0 and x1<=Wf and y1<=Hf:
                        a=pch[...,3]/255.0
                        for c in range(3): img[y0:y1,x0:x1,c]=a*pch[...,c]+(1-a)*img[y0:y1,x0:x1,c]
                # выбор папки
                if random.random()<TRAIN_RATIO:
                    out_img, out_lbl = TRAIN_IMG_DIR, TRAIN_LBL_DIR
                else:
                    out_img, out_lbl = VAL_IMG_DIR, VAL_LBL_DIR
                fname = f"{k}_{'-'.join(combo)}_{idx}.jpg"
                cv2.imwrite(str(out_img/fname), img)
                with open(out_lbl/fname.replace('.jpg','.txt'),'w') as f:
                    f.write("\n".join(ann))
                metadata.append((out_img/fname, ann))
                pbar.update(1)
                break
pbar.close()

# data.yaml
cfg={"train":str(TRAIN_IMG_DIR),"val":str(VAL_IMG_DIR),"nc":len(CLASS_NAMES),"names":{i:n for i,n in enumerate(CLASS_NAMES)}}
with open(TRAIN_DATA/"data.yaml","w") as f: yaml.dump(cfg,f,sort_keys=False)

# QA-примеры
qa = random.sample(metadata, min(15,len(metadata)))
for img_path, ann in qa:
    img = cv2.imread(str(img_path)); H,W=img.shape[:2]
    for line in ann:
        cid,xc,yc,bw,bh = map(float,line.split())
        x0=int((xc-bw/2)*W); y0=int((yc-bh/2)*H)
        x1=int((xc+bw/2)*W); y1=int((yc+bh/2)*H)
        cv2.rectangle(img,(x0,y0),(x1,y1),(0,255,0),2)
    cv2.imwrite(str(QA_DIR/img_path.name), img)

print("Генерация завершена!")
