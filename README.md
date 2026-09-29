# Onion Quality Model — friend package

YOLOv8n onion condition detector (Healthy / Rotten / Sprouted / Damaged),
25 epochs, mAP50 0.58, condition accuracy ~73%. Model file is only ~6 MB.

## Contents
- `model/best.pt` — trained weights (copy of a finished run, nothing else needed)
- `config.yaml` — size limits (35/70 mm), quality threshold (75), camera `pixels_per_mm`
- `grader.py` — CHOOSE / DO NOT CHOOSE decision layer (do NOT decide on YOLO confidence alone)
- `size_estimator.py` — pixels → mm using your calibrated `pixels_per_mm`
- `predict.py` — minimal working example

## Use in your Streamlit app
```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown by Streamlit and choose **Live camera**. Allow browser
access, then choose **Stop camera & create report** to finish the inspection and
download a PDF/QR report of detected onions. Frames are analyzed live and are
not recorded or saved. Choose **Upload** to select and inspect multiple photos
in one batch; only annotated detection results are shown. Camera access requires
localhost or HTTPS; remote HTTP pages cannot use the browser camera.

In the sidebar's **Fixed camera ruler calibration** expander, enter a known
ruler span in mm and the same span's length in pixels from the fixed camera
image. Apply it once; the resulting `pixels_per_mm` and confirmation are saved
to `config.yaml`. Recalibrate if the camera or its height changes.

The detector predicts condition classes; diameter and the Small / Medium / Large
/ Oversize size grade are calculated from the onion bounding box and calibrated
`pixels_per_mm`. The quality letter grade is derived from the numeric score:
A+ (95+), A (90-94), B+ (85-89), B (75-84), Below B (<75). Only Medium/Large
size grades and A+/A/B+/B quality grades are accepted by default, along with the
condition and confidence rules. These score bands and accepted grades are
configurable. The QR code contains a readable text summary, not JSON.

For a standalone image prediction:
```bash
python predict.py onion.jpg
```
```python
from ultralytics import YOLO
model = YOLO("model/best.pt")   # adjust path to where you put this folder
res = model.predict("onion.jpg", conf=0.35, verbose=False)[0]
```

## Important
1. **Calibrate once:** set `calibration.pixels_per_mm` in `config.yaml` for YOUR camera
   (photo a ruler once: `pixels_per_mm = object_px / object_mm`). Default 4.35 is a placeholder.
2. Model labels: `healthy_onion, rotten_onion, sprouted_onion, damaged_onion, sprout, reference`.
   `reference` is a detector label, not an onion condition or size grade.
3. Feed every detection through `grader.decide()` — it combines size + condition +
   defects + quality score into the final CHOOSE / DO NOT CHOOSE.
