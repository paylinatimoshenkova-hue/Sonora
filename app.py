
import os
import uuid
import subprocess
import tempfile

from pathlib import Path
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import librosa
import imageio_ffmpeg


app = Flask(__name__)

BASE = Path(__file__).parent
LIBRARY = BASE / "library"
LIBRARY.mkdir(exist_ok=True)

# Максимальный размер одного запроса — 100 МБ
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

# Поддерживаемые форматы
ALLOWED = {
    ".mp3",
    ".wav",
    ".m4a",
    ".aac",
    ".ogg",
    ".flac",
    ".webm",
    ".mp4"
}

SAMPLE_RATE = 22050


def convert_to_wav(source_path):
    """
    Конвертирует MP3 и другие аудиоформаты в WAV
    с помощью FFmpeg.
    FFmpeg устанавливается вместе с imageio-ffmpeg.
    """

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    temp = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    )

    wav_path = temp.name
    temp.close()

    command = [
        ffmpeg,
        "-y",
        "-i", str(source_path),
        "-vn",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-f", "wav",
        wav_path
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180
        )

        if result.returncode != 0:
            error = result.stderr.decode(
                "utf-8",
                errors="ignore"
            )
            raise ValueError(
                "Не удалось декодировать аудио: " + error[-500:]
            )

        return wav_path

    except Exception:
        Path(wav_path).unlink(missing_ok=True)
        raise


def fingerprint(path):
    """
    Создаёт звуковой отпечаток композиции.
    """

    wav_path = convert_to_wav(path)

    try:
        y, sr = librosa.load(
            wav_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        if len(y) < sr:
            raise ValueError(
                "Аудиофайл слишком короткий. Нужно минимум 1 секунду."
            )

        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=sr,
            hop_length=512
        )

        profile = np.mean(chroma, axis=1)

        return profile

    finally:
        Path(wav_path).unlink(missing_ok=True)


def similarity(a, b):
    """
    Сравнивает два звуковых отпечатка.
    """

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    score = np.dot(a, b) / denominator

    return float(np.clip(score, 0, 1))


@app.get("/")
def index():
    tracks = sorted(
        p.name
        for p in LIBRARY.iterdir()
        if p.is_file()
    )

    return render_template(
        "index.html",
        tracks=tracks
    )


@app.post("/upload")
def upload():
    """
    Загружает несколько музыкальных композиций.
    """

    items = request.files.getlist("files")

    if not items:
        return jsonify({
            "error": "Ты не выбрал файлы."
        }), 400

    saved = []
    errors = []

    for item in items:

        if not item or not item.filename:
            continue

        original_name = secure_filename(item.filename)

        if not original_name:
            errors.append("Некорректное имя файла.")
            continue

        ext = Path(original_name).suffix.lower()

        if ext not in ALLOWED:
            errors.append(
                f"{original_name}: формат не поддерживается."
            )
            continue

        unique_name = f"{uuid.uuid4().hex[:10]}_{original_name}"
        target = LIBRARY / unique_name

        try:
            item.save(target)

            if not target.exists() or target.stat().st_size == 0:
                raise ValueError("Файл пустой.")

            saved.append(unique_name)

        except Exception as e:
            target.unlink(missing_ok=True)
            errors.append(f"{original_name}: {str(e)}")

    return jsonify({
        "saved": saved,
        "count": len(saved),
        "errors": errors
    })


@app.post("/match")
def match():
    """
    Принимает запись с микрофона и ищет
    наиболее похожие композиции в библиотеке.
    """

    clip = request.files.get("audio")

    if not clip or not clip.filename:
        return jsonify({
            "error": "Сначала запиши фрагмент."
        }), 400

    ext = Path(clip.filename).suffix.lower()

    if ext not in ALLOWED:
        return jsonify({
            "error": "Формат записи не поддерживается."
        }), 400

    temp_path = BASE / f"clip_{uuid.uuid4().hex}{ext}"

    try:
        clip.save(temp_path)

        query = fingerprint(temp_path)

        results = []

        for path in LIBRARY.iterdir():

            if not path.is_file():
                continue

            if path.suffix.lower() not in ALLOWED:
                continue

            try:
                track_profile = fingerprint(path)

                score = similarity(
                    query,
                    track_profile
                )

                results.append({
                    "name": path.name,
                    "score": round(score * 100, 1)
                })

            except Exception as e:
                print(
                    f"Ошибка обработки {path.name}: {e}"
                )

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return jsonify({
            "results": results[:5]
        })

    except Exception as e:

        print("Ошибка поиска:", str(e))

        return jsonify({
            "error": (
                "Не удалось обработать аудио. "
                "Попробуй другой файл или запись. "
                + str(e)
            )
        }), 400

    finally:
        temp_path.unlink(missing_ok=True)


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "Файл слишком большой. Максимальный размер — 100 МБ."
    }), 413


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
