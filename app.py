
import os
import json
import uuid
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import librosa
import imageio_ffmpeg

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from scipy.ndimage import maximum_filter


app = Flask(__name__)

BASE = Path(__file__).parent
LIBRARY = BASE / "library"
INDEXES = BASE / "fingerprints"

LIBRARY.mkdir(exist_ok=True)
INDEXES.mkdir(exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

ALLOWED = {
    ".mp3", ".wav", ".m4a", ".aac",
    ".ogg", ".flac", ".webm", ".mp4"
}

SR = 11025
HOP = 256
N_FFT = 2048


def convert_audio(source):
    """Преобразует аудио в WAV для анализа."""

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    temp = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    )
    output = temp.name
    temp.close()

    command = [
        ffmpeg, "-y",
        "-i", str(source),
        "-vn",
        "-ac", "1",
        "-ar", str(SR),
        "-f", "wav",
        output
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180
        )

        if result.returncode != 0:
            raise ValueError(
                result.stderr.decode(
                    "utf-8", errors="ignore"
                )[-500:]
            )

        return output

    except Exception:
        Path(output).unlink(missing_ok=True)
        raise


def make_fingerprint(source):
    """
    Создаёт звуковые отпечатки из пар
    спектральных пиков.
    """

    wav = convert_audio(source)

    try:
        audio, sr = librosa.load(
            wav,
            sr=SR,
            mono=True
        )

        if len(audio) < sr:
            raise ValueError(
                "Аудио должно быть длиннее одной секунды."
            )

        spectrum = np.abs(
            librosa.stft(
                audio,
                n_fft=N_FFT,
                hop_length=HOP
            )
        )

        db = librosa.amplitude_to_db(
            spectrum,
            ref=np.max
        )

        local_max = maximum_filter(
            db,
            size=(9, 9)
        )

        mask = (
            (db == local_max) &
            (db > -45)
        )

        freq_bins, frames = np.where(mask)

        # Убираем очень низкие частоты
        selected = [
            (int(f), int(t))
            for f, t in zip(freq_bins, frames)
            if f >= 5
        ]

        # Ограничиваем количество пиков
        selected.sort(key=lambda x: x[1])

        peaks_by_time = defaultdict(list)

        for freq, frame in selected:
            peaks_by_time[frame].append(freq)

        peaks = []

        for frame, freqs in peaks_by_time.items():
            for freq in freqs[:5]:
                peaks.append((freq, frame))

        peaks.sort(key=lambda x: x[1])

        fingerprints = defaultdict(list)

        for i, (f1, t1) in enumerate(peaks):

            future = []

            for f2, t2 in peaks[i + 1:]:

                delta = t2 - t1

                if delta > 10:
                    break

                if delta >= 1:
                    future.append((f2, delta))

            for f2, delta in future[:5]:

                key = f"{f1}:{f2}:{delta}"

                time_seconds = round(
                    t1 * HOP / SR,
                    3
                )

                fingerprints[key].append(
                    time_seconds
                )

        return dict(fingerprints)

    finally:
        Path(wav).unlink(missing_ok=True)


def get_track_index(track):
    """Загружает сохранённый индекс или создаёт новый."""

    index_file = INDEXES / (track.name + ".json")

    modified = track.stat().st_mtime

    if index_file.exists():

        try:
            data = json.loads(
                index_file.read_text(encoding="utf-8")
            )

            if data.get("modified") == modified:
                return data["fingerprints"]

        except Exception:
            pass

    fingerprints = make_fingerprint(track)

    data = {
        "modified": modified,
        "fingerprints": fingerprints
    }

    index_file.write_text(
        json.dumps(data),
        encoding="utf-8"
    )

    return fingerprints


@app.get("/")
def index():

    tracks = sorted(
        p.name for p in LIBRARY.iterdir()
        if p.is_file() and p.suffix.lower() in ALLOWED
    )

    return render_template(
        "index.html",
        tracks=tracks
    )


@app.post("/upload")
def upload():

    files = request.files.getlist("files")

    if not files:
        return jsonify({
            "error": "Файлы не выбраны."
        }), 400

    saved = []
    errors = []

    for file in files:

        if not file or not file.filename:
            continue

        name = secure_filename(file.filename)
        ext = Path(name).suffix.lower()

        if ext not in ALLOWED:
            errors.append(
                f"{name}: формат не поддерживается."
            )
            continue

        unique_name = (
            uuid.uuid4().hex[:8] + "_" + name
        )

        target = LIBRARY / unique_name

        try:

            file.save(target)

            if not target.exists() or target.stat().st_size == 0:
                raise ValueError("Файл пустой.")

            saved.append(unique_name)

        except Exception as e:

            target.unlink(missing_ok=True)
            errors.append(f"{name}: {str(e)}")

    return jsonify({
        "saved": saved,
        "count": len(saved),
        "errors": errors
    })


@app.post("/match")
def match():

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

    temp = BASE / (
        "clip_" + uuid.uuid4().hex + ext
    )

    try:

        clip.save(temp)

        query = make_fingerprint(temp)

        query_count = sum(
            len(times) for times in query.values()
        )

        if query_count < 10:
            return jsonify({
                "error": (
                    "Слишком мало звуковых данных. "
                    "Запиши фрагмент подольше."
                )
            }), 400

        results = []

        for track in LIBRARY.iterdir():

            if not track.is_file():
                continue

            if track.suffix.lower() not in ALLOWED:
                continue

            try:

                reference = get_track_index(track)

                offsets = []

                for key, query_times in query.items():

                    reference_times = reference.get(key, [])

                    for qt in query_times:
                        for rt in reference_times:
                            offsets.append(round(rt - qt, 1))

                if not offsets:
                    continue

                histogram = Counter(offsets)

                best_offset, matches = histogram.most_common(1)[0]

                score = round(
                    matches / max(query_count, 1) * 100,
                    1
                )

                if matches >= 5:

                    results.append({
                        "name": track.name,
                        "score": min(score, 100),
                        "position": max(0, best_offset)
                    })

            except Exception as e:

                print(
                    "Ошибка обработки",
                    track.name,
                    str(e)
                )

        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        if not results:
            return jsonify({
                "results": [],
                "message": "Совпадение не найдено."
            })

        return jsonify({
            "results": results[:5]
        })

    except Exception as e:

        print("Ошибка поиска:", str(e))

        return jsonify({
            "error": "Ошибка обработки аудио: " + str(e)
        }), 400

    finally:
        temp.unlink(missing_ok=True)


@app.errorhandler(413)
def too_large(error):

    return jsonify({
        "error": "Файл слишком большой. Максимум 100 МБ."
    }), 413


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
