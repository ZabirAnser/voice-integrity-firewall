import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ml_inference import evaluate_audio_authenticity

app = FastAPI(title="Voice Security API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    allowed_extensions = (".wav", ".mp3", ".m4a", ".ogg")
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(status_code=400, detail="Invalid audio format")

    suffix = os.path.splitext(file.filename)[1]
    
    # Write incoming bytes to a temporary file for librosa
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(await file.read())
        temp_audio_path = temp_audio.name

    try:
        # Run RawNet3 inference
        spoof_probability = evaluate_audio_authenticity(temp_audio_path)
        fake_percentage = round(spoof_probability * 100.0, 2)

        # Risk scoring
        if fake_percentage >= 75.0:
            risk_level = "High"
            action = "Immediate threat: Synthesized audio signature detected. Halt transactions."
        elif fake_percentage >= 40.0:
            risk_level = "Medium"
            action = "Suspicious vocal anomalies detected. Request secondary verification."
        else:
            risk_level = "Low"
            action = "Natural acoustic markers confirmed. Authentic human voice."

        return {
            "filename": file.filename,
            "risk_level": risk_level,
            "ml_score_percent": fake_percentage,
            "recommendation": action
        }

    finally:
        # Clean up temporary audio file from disk
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
