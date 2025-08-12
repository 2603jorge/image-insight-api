import io
import os
import base64
from datetime import datetime

from flask import Flask, request, jsonify
from PIL import Image, ExifTags, ImageOps
import pytesseract
import numpy as np
import cv2
from pyzbar import pyzbar

app = Flask(__name__)

def load_image_from_request() -> Image.Image:
    if "file" in request.files:
        file = request.files["file"]
        return Image.open(file.stream).convert("RGB")
    data = request.get_json(silent=True) or {}
    b64 = data.get("image_base64")
    if not b64:
        raise ValueError("Falta la imagen (sube 'file' o 'image_base64').")
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")

def image_basic_info(img: Image.Image):
    w, h = img.size
    return {"width": w, "height": h, "mode": img.mode, "created_at": datetime.utcnow().isoformat() + "Z"}

def image_exif(img: Image.Image):
    try:
        exif = img.getexif()
        if not exif:
            return {}
        readable = {}
        for k, v in exif.items():
            key = ExifTags.TAGS.get(k, str(k))
            if isinstance(v, bytes):
                try:
                    v = v.decode("utf-8", errors="ignore")
                except Exception:
                    v = str(v)
            readable[key] = v
        return readable
    except Exception:
        return {}

def fix_orientation(img: Image.Image) -> Image.Image:
    try:
        return ImageOps.exif_transpose(img)
    except Exception:
        return img

def ocr_text(img: Image.Image, lang: str = "eng+spa"):
    try:
        scale = 1.5
        up = img.resize((int(img.width * scale), int(img.height * scale)))
        text = pytesseract.image_to_string(up, lang=lang)
        return text.strip()
    except Exception as e:
        return f"(OCR error: {e})"

def detect_barcodes(img: Image.Image):
    try:
        arr = np.array(img)[:, :, ::-1]
        decoded = pyzbar.decode(arr)
        return [{"type": d.type, "data": d.data.decode("utf-8", errors="ignore"), "rect": {"x": d.rect.left, "y": d.rect.top, "w": d.rect.width, "h": d.rect.height}} for d in decoded]
    except Exception:
        return []

def detect_faces(img: Image.Image):
    try:
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        return {"count": len(faces), "boxes": [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for (x, y, w, h) in faces]}
    except Exception as e:
        return {"count": 0, "error": str(e), "boxes": []}

@app.post("/analyze")
def analyze():
    try:
        img = load_image_from_request()
        img = fix_orientation(img)
        return jsonify({
            "ok": True,
            "info": image_basic_info(img),
            "exif": image_exif(img),
            "ocr_text": ocr_text(img, lang=request.args.get("lang", "eng+spa")),
            "barcodes": detect_barcodes(img),
            "faces": detect_faces(img)
        }), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.get("/health")
def health():
    return "ok", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))