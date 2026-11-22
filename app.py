from flask import Flask, render_template, request, jsonify
from pathlib import Path
import numpy as np
import librosa
import uuid

app = Flask(__name__)
BASE = Path(__file__).parent
LIBRARY = BASE / "library"
LIBRARY.mkdir(exist_ok=True)
ALLOWED = {".wav", ".mp3", ".m
