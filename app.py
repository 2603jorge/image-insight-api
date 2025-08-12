import io
import os
import base64
from datetime import datetime

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from PIL import Image, ExifTags, ImageOps
import pytesseract
import numpy as np
import cv2
from pyzbar import pyzbar

app = Flask(__name__)
# Habilitamos CORS por si usas el cliente local, pero al servir /client no es necesario
CORS(app, resources={r"/*": {"origins": "*"}})

# ---------- Utilidades ----------
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
        return [{
            "type": d.type,
            "data": d.data.decode("utf-8", errors="ignore"),
            "rect": {"x": d.rect.left, "y": d.rect.top, "w": d.rect.width, "h": d.rect.height}
        } for d in decoded]
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

# ---------- Endpoints ----------
@app.get("/")
def home():
    return "API funcionando correctamente", 200

@app.get("/health")
def health():
    return "ok", 200

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

# Cliente web servido por la API (misma-origin, sin CORS)
CLIENT_HTML = """<!DOCTYPE html>
<html lang="es"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cliente Web • Image Insight API</title>
<style>
body{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;background:#0b1220;color:#e5e7eb;margin:0;padding:24px}
.container{max-width:900px;margin:0 auto}
.card{background:linear-gradient(180deg,#0f172a 0%,#0b1020 100%);border:1px solid #1f2937;border-radius:16px;padding:20px}
h1{margin:0 0 12px;font-size:24px}
label{display:block;margin:8px 0 6px}
input,select,button{background:#0b1220;border:1px solid #374151;color:#e5e7eb;padding:10px 12px;border-radius:10px}
button.primary{background:#22c55e;border:0;color:#052e16;font-weight:600;cursor:pointer}
.preview{background:#020617;border:1px dashed #374151;border-radius:12px;min-height:220px;display:flex;align-items:center;justify-content:center;overflow:hidden;margin-top:8px}
.preview img{max-width:100%}
pre{background:#020617;border:1px solid #1f2937;border-radius:12px;padding:12px;overflow:auto;max-height:360px}
.small{color:#9ca3af;font-size:12px}
</style></head>
<body><div class="container"><div class="card">
<h1>Cliente Web • Image Insight API</h1>
<div style="display:flex;gap:16px;flex-wrap:wrap">
  <div style="flex:1 1 320px">
    <label for="lang" class="small">Idioma OCR</label>
    <select id="lang"><option value="eng">inglés (eng)</option><option value="spa" selected>español (spa)</option><option value="eng+spa">inglés+español</option></select>
    <label for="mode" class="small">Modo de envío</label>
    <select id="mode"><option value="file" selected>Archivo (multipart/form-data)</option><option value="base64">Base64 (JSON)</option></select>
    <label for="fileInput">Selecciona una imagen</label>
    <input id="fileInput" type="file" accept="image/*">
    <div style="margin-top:12px;display:flex;gap:8px">
      <button class="primary" id="sendBtn">Enviar a la API</button>
      <button id="clearBtn">Limpiar</button>
    </div>
  </div>
  <div style="flex:1 1 320px">
    <div class="preview" id="previewBox"><span class="small">Vista previa</span></div>
  </div>
</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:16px">
  <div><h3>Respuesta</h3><pre id="respBox">{}</pre></div>
  <div><h3>Texto OCR</h3><pre id="ocrBox"></pre></div>
</div>
</div></div>
<script>
const fileInput=document.getElementById('fileInput');
const previewBox=document.getElementById('previewBox');
const sendBtn=document.getElementById('sendBtn');
const clearBtn=document.getElementById('clearBtn');
const respBox=document.getElementById('respBox');
const ocrBox=document.getElementById('ocrBox');
const lang=document.getElementById('lang');
const mode=document.getElementById('mode');
let currentImageB64=null;
fileInput.addEventListener('change',()=>{const f=fileInput.files[0];if(!f)return;const r=new FileReader();r.onload=e=>{const url=e.target.result;currentImageB64=url;previewBox.innerHTML=`<img src="${url}" alt="preview">`};r.readAsDataURL(f);});
sendBtn.addEventListener('click',async()=>{if(!fileInput.files[0]){alert('Selecciona una imagen');return;}respBox.textContent='Enviando...';ocrBox.textContent='';try{let r;if(mode.value==='file'){const fd=new FormData();fd.append('file',fileInput.files[0]);r=await fetch(`/analyze?lang=${encodeURIComponent(lang.value)}`,{method:'POST',body:fd});}else{r=await fetch(`/analyze?lang=${encodeURIComponent(lang.value)}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image_base64:currentImageB64})});}const data=await r.json();respBox.textContent=JSON.stringify(data,null,2);ocrBox.textContent=(data&&data.ocr_text)?data.ocr_text:'';}catch(e){respBox.textContent='Error: '+e.message;}});
clearBtn.addEventListener('click',()=>{fileInput.value='';currentImageB64=null;previewBox.innerHTML='<span class="small">Vista previa</span>';respBox.textContent='{}';ocrBox.textContent='';});
</script></body></html>
"""

@app.get("/client")
def client():
    return Response(CLIENT_HTML, mimetype="text/html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

