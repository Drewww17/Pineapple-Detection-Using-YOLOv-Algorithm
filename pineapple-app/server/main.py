import os, io, csv, datetime
from typing import Dict, Any, List

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from PIL import Image
from ultralytics import YOLO

BASE_DIR = os.path.dirname(__file__)
WEB_DIR = os.path.join(BASE_DIR, "..", "web")
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

YOLOV8_WEIGHTS  = os.getenv("YOLOV8_WEIGHTS", "./yolov8_best.pt")
YOLOV11_WEIGHTS = os.getenv("YOLOV11_WEIGHTS", "./yolov11_best.pt")
YOLOV12_WEIGHTS = os.getenv("YOLOV12_WEIGHTS", "./yolov12_best.pt")

CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
IOU_THRESHOLD  = float(os.getenv("IOU_THRESHOLD",  "0.45"))
IMG_SIZE       = int(float(os.getenv("IMG_SIZE", "512")))

DEVICE = "cpu"

app = FastAPI(title="Pineapple Ripeness – Local YOLOv8/v11/v12")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

LOG_PATH = os.path.join(BASE_DIR, "logs.csv")

def ensure_log_header():
    if not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0:
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp",
                "v8_label","v8_conf",
                "v11_label","v11_conf",
                "v12_label","v12_conf",
                "verdict_label","verdict_conf"
            ])

def append_log(v8: Dict[str, Any], v11: Dict[str, Any], v12: Dict[str, Any], verdict: Dict[str, Any]):
    ensure_log_header()
    row = [
        datetime.datetime.now().isoformat(timespec="seconds"),
        (v8.get("summary",{}).get("class") if v8.get("ok") else ""), (v8.get("summary",{}).get("confidence") if v8.get("ok") else ""),
        (v11.get("summary",{}).get("class") if v11.get("ok") else ""), (v11.get("summary",{}).get("confidence") if v11.get("ok") else ""),
        (v12.get("summary",{}).get("class") if v12.get("ok") else ""), (v12.get("summary",{}).get("confidence") if v12.get("ok") else ""),
        verdict.get("label",""), verdict.get("confidence","")
    ]
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

def ensemble_vote(results: Dict[str, Any]) -> Dict[str, Any]:
    picks = []
    for k in ("YOLOv8","YOLOv11","YOLOv12"):
        r = results.get(k, {})
        if r.get("ok") and r.get("summary"):
            picks.append((k, r["summary"]["class"], float(r["summary"]["confidence"])))
    if not picks:
        return {"label":"No detection", "confidence":0.0}
    from collections import Counter
    c = Counter([lab for _, lab, _ in picks])
    top = max(c.values())
    finalists = [lab for lab, ct in c.items() if ct == top]
    if len(finalists) == 1:
        label = finalists[0]
        conf = max(conf for _, lab, conf in picks if lab == label)
        return {"label": label, "confidence": conf}
    best = max((p for p in picks if p[1] in finalists), key=lambda t: t[2])
    return {"label": best[1], "confidence": best[2]}

_models: Dict[str, YOLO] = {}

def _load(path: str) -> YOLO:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return YOLO(path)

def get_model(key: str, path: str) -> YOLO:
    if key not in _models:
        _models[key] = _load(path)
    return _models[key]

def y_predict(model: YOLO, img: Image.Image) -> List[Dict[str, Any]]:
    res = model.predict(
        img,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=DEVICE,   
        verbose=False
    )
    out: List[Dict[str, Any]] = []
    if not res:
        return out
    r0 = res[0]
    names = getattr(r0, "names", {})
    xyxy = r0.boxes.xyxy.cpu().tolist() if hasattr(r0.boxes, "xyxy") else []
    conf = r0.boxes.conf.cpu().tolist() if hasattr(r0.boxes, "conf") else []
    cls  = r0.boxes.cls.cpu().tolist() if hasattr(r0.boxes, "cls") else []
    for (x1,y1,x2,y2), c, k in zip(xyxy, conf, cls):
        w, h = (x2-x1), (y2-y1)
        cx, cy = x1 + w/2, y1 + h/2
        label = names.get(int(k), str(int(k)))
        out.append({
            "x": float(cx), "y": float(cy),
            "width": float(w), "height": float(h),
            "class": label, "confidence": float(c)
        })
    return out

def summarize(preds: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not preds:
        return {"class": "No detection", "confidence": 0.0}
    best = max(preds, key=lambda p: p.get("confidence", 0.0))
    return {"class": best.get("class","Unknown"), "confidence": float(best.get("confidence", 0.0))}

@app.get("/")
def root():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/logs.csv")
def download_logs():
    ensure_log_header()
    return FileResponse(LOG_PATH, media_type="text/csv", filename="pineapple_logs.csv")

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    out: Dict[str, Any] = {}

    try:
        m8 = get_model("y8", YOLOV8_WEIGHTS)
        p8 = y_predict(m8, img)
        out["YOLOv8"] = {"ok": True, "model": YOLOV8_WEIGHTS, "predictions": p8, "summary": summarize(p8)}
    except Exception as e:
        out["YOLOv8"] = {"ok": False, "error": str(e), "model": YOLOV8_WEIGHTS}

    try:
        m11 = get_model("y11", YOLOV11_WEIGHTS)
        p11 = y_predict(m11, img)
        out["YOLOv11"] = {"ok": True, "model": YOLOV11_WEIGHTS, "predictions": p11, "summary": summarize(p11)}
    except Exception as e:
        out["YOLOv11"] = {"ok": False, "error": str(e), "model": YOLOV11_WEIGHTS}

    try:
        m12 = get_model("y12", YOLOV12_WEIGHTS)
        p12 = y_predict(m12, img)
        out["YOLOv12"] = {"ok": True, "model": YOLOV12_WEIGHTS, "predictions": p12, "summary": summarize(p12)}
    except Exception as e:
        out["YOLOv12"] = {"ok": False, "error": str(e), "model": YOLOV12_WEIGHTS}

    out["verdict"] = ensemble_vote(out)
    append_log(out.get("YOLOv8", {}), out.get("YOLOv11", {}), out.get("YOLOv12", {}), out["verdict"])
    return JSONResponse(out)
