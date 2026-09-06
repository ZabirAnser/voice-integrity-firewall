from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
import soundfile as sf
import numpy as np

app = FastAPI(title="Voice Security API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (good for hackathon testing)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)

# ==========================================
# 1. PLACEHOLDER ML FUNCTION
# ==========================================
def predict_voice_authenticity(audio_array: np.ndarray) -> float:
    """
    Member 2 will replace this function later.
    For now, it acts as a dummy model returning a fake probability (e.g., 0.85 = 85% fake)
    """
    return 0.85 


@app.post("/api/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    allowed_extensions = (".wav", ".mp3", ".m4a", ".ogg")
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Please upload an audio file like {allowed_extensions}."
        )

    try:
        contents = await file.read()
        audio_stream = io.BytesIO(contents)
        audio_array, sample_rate = sf.read(audio_stream)

        
        # ==========================================
        # 2. AUDIO PREPROCESSING (Stereo to Mono)
        # ==========================================
        if len(audio_array.shape) > 1:
            # If the audio has multiple channels (stereo), average them into one (mono)
            audio_array = np.mean(audio_array, axis=1)

        # ==========================================
        # 3. GET ML PREDICTION
        # ==========================================
        fake_probability = predict_voice_authenticity(audio_array)
        fake_percentage = round(fake_probability * 100, 1)

        # ==========================================
        # 4. CALCULATE RISK SCORE
        # ==========================================
        if fake_percentage >= 75.0:
            risk_level = "High risk"
            summary = "Voice may be AI-generated/manipulated."
            recommendation = "Warn the user and recommend stopping or holding the transaction."
        elif fake_percentage >= 40.0:
            risk_level = "Medium risk"
            summary = "Minor synthetic anomalies detected."
            recommendation = "Ask for additional verification."
        else:
            risk_level = "Low risk"
            summary = "Call seems normal."
            recommendation = "Audio appears genuine."

        # Return the exact JSON structure the frontend needs
        return {
            "status": "success",
            "filename": file.filename,
            "duration_seconds": round(len(audio_array) / sample_rate, 2),
            "ml_score_percent": fake_percentage,
            "risk_level": risk_level,
            "summary": summary,
            "recommendation": recommendation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")