from flask import Flask, render_template, request, jsonify
from pathlib import Path
import numpy as np
import librosa
import uuid

app = Flask(__name__)
BASE = Path(__file__).parent
LIBRARY = BASE / "library"
LIBRARY.mkdir(exist_ok=True)
ALLOWED = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".webm"}

def fingerprint(path):
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    if len(y) < sr:
        raise ValueError("Файл слишком короткий")
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    # Усреднённый профиль тональных классов — простой демонстрационный отпечаток.
    return np.mean(chroma, axis=1)

def similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0

@app.get("/")
def index():
    tracks = sorted(p.name for p in LIBRARY.iterdir() if p.is_file())
    return render_template("index.html", tracks=tracks)

@app.post("/upload")
def upload():
    items = request.files.getlist("files")
    saved = []
    for item in items:
        if not item.filename:
            continue
        ext = Path(item.filename).suffix.lower()
        if ext not in ALLOWED:
            continue
        safe_name = Path(item.filename).name
        target = LIBRARY / safe_name
        if target.exists():
            target = LIBRARY / f"{uuid.uuid4().hex[:8]}_{safe_name}"
        item.save(target)
        saved.append(target.name)
    return jsonify({"saved": saved, "count": len(saved)})

@app.post("/match")
def match():
    clip = request.files.get("audio")
    if not clip:
        return jsonify({"error": "Сначала запишите фрагмент."}), 400
    temp = BASE / f"clip_{uuid.uuid4().hex}.webm"
    clip.save(temp)
    try:
        query = fingerprint(temp)
        results = []
        for path in LIBRARY.iterdir():
            if path.is_file() and path.suffix.lower() in ALLOWED:
                try:
                    score = similarity(query, fingerprint(path))
                    results.append({"name": path.name, "score": round(score * 100, 1)})
                except Exception:
                    pass
        results.sort(key=lambda x: x["score"], reverse=True)
        return jsonify({"results": results[:5]})
    except Exception as e:
        return jsonify({"error": f"Не удалось обработать запись: {e}. Проверьте FFmpeg и формат файла."}), 400
    finally:
        temp.unlink(missing_ok=True)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
