"""Minimal onion prediction example. Run: python predict.py onion.jpg"""
import sys, yaml
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from grader import OnionInput, decide, format_report
from size_estimator import px_diameter_from_bbox, onion_diameter_mm

CLASSES = ['healthy_onion', 'rotten_onion', 'sprouted_onion', 'damaged_onion', 'sprout']

def classify(label: str):
    ll = label.lower()
    if "rotten" in ll or "mold" in ll: return "Rotten", {"rot": True}
    if "sprout" in ll or "root" in ll or "stem" in ll: return "Sprouted", {"sprout": True}
    if "damage" in ll or "cut" in ll: return "Damaged", {"damage": True}
    return "Healthy", {}

def main():
    from ultralytics import YOLO
    src = sys.argv[1] if len(sys.argv) > 1 else "onion.jpg"
    cfg = yaml.safe_load(open(HERE / "config.yaml"))
    ppm = float(cfg["calibration"]["pixels_per_mm"])  # calibrate once for your camera!
    model = YOLO(str(HERE / "model" / "best.pt"))
    res = model.predict(src, conf=0.35, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        print("No onions detected."); return
    for box in res.boxes:
        cls, c = int(box.cls.item()), float(box.conf.item())
        cond, flags = classify(res.names[cls])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        mm = onion_diameter_mm(px_diameter_from_bbox(abs(x2 - x1), abs(y2 - y1)), ppm)
        print(format_report(decide(OnionInput(mm, cond, confidence=c, **flags), cfg)))
        print("-" * 30)

if __name__ == "__main__":
    main()
