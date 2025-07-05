import numpy as np
import argparse
import cv2
import torch
import clip
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# ─── Аргументы ───
parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, required=True, help="Путь к видеофайлу")
parser.add_argument("--output", type=str, default=None, help="Путь к выходному видео")
parser.add_argument("--conf", type=float, default=0.6, help="Порог confidence для YOLO")
parser.add_argument("--iou", type=float, default=0.55, help="Порог IoU для YOLO")
args = parser.parse_args()

# ───────── ПАРАМЕТРЫ ─────────
VIDEO_IN = args.video
VIDEO_OUT = args.output or (str(Path(VIDEO_IN).with_name(Path(VIDEO_IN).stem + "_out.mp4")))
YOLO_WEIGHTS   = "weights/model.pt"
IMGSZ          = 640
CONF = args.conf
IOU = args.iou
CLIP_LABELS    = ["dish is empty", "dish with food"]
COLOR_FULL     = (0,255,0)
COLOR_EMPTY    = (0,0,255)


device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(device)
yolo = YOLO(YOLO_WEIGHTS).to(device).eval()
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
text_tokens = clip.tokenize(CLIP_LABELS).to(device)

cap = cv2.VideoCapture(VIDEO_IN)

meta_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
meta_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

ret, frame = cap.read()
frame_h, frame_w = frame.shape[:2]
rot_needed = (meta_h > meta_w) and (frame_w > frame_h)
if not rot_needed:
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

h, w = frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (w, h))

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
pbar = tqdm(total=total_frames, desc="Frames")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if not rot_needed:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    results = yolo(frame, imgsz=IMGSZ, conf=CONF, iou=IOU)[0]
    crops = []
    for box in results.boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
        dx, dy = int((x2-x1)*0.05), int((y2-y1)*0.05)
        x1n, y1n = max(0,x1-dx), max(0,y1-dy)
        x2n, y2n = min(w,x2+dx), min(h,y2+dy)
        crop = frame[y1n:y2n, x1n:x2n]
        if crop.size == 0: continue
        crops.append((crop, (x1n, y1n, x2n, y2n)))

    if crops:
        clip_inputs = torch.stack([
            clip_preprocess(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))).to(device)
            for c,_ in crops
        ])
        with torch.no_grad():
            image_features = clip_model.encode_image(clip_inputs)
            text_features  = clip_model.encode_text(text_tokens)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features  /= text_features.norm(dim=-1, keepdim=True)
            probs = (image_features @ text_features.T).softmax(dim=-1)  # [N,2]

        for i,(_,box) in enumerate(crops):
            x1,y1,x2,y2 = box
            score_empty, score_full = probs[i].cpu().tolist()
            orig_cls = int(results.boxes.cls[i].item())
            cls_name = yolo.names[orig_cls]
            if score_empty > score_full:
                label = f"{cls_name}_empty ({score_empty:.2f})"
                clr   = COLOR_EMPTY
            else:
                conf     = float(results.boxes.conf[i].item())
                label = f"{cls_name} {conf:.2f}"
                clr   = COLOR_FULL
            cv2.rectangle(frame, (x1,y1), (x2,y2), clr, 2)
            cv2.putText(frame, label, (x1, max(0,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 3, clr, 3)

    out.write(frame)
    pbar.update(1)

cap.release()
out.release()
pbar.close()
print(f"saved in {VIDEO_OUT}")