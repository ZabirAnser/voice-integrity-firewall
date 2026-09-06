import os
import sys
import torch
import torch.nn as nn
import numpy as np
import importlib.util
import librosa

# 1. Base Paths Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Adjust relative path if this file is placed inside backend/
MODEL_PATH = os.path.join(BASE_DIR, "python/RawNet3/models/weights/model.pt")

# 2. Locate and import RawNet3 architecture
rawnet3_script_path = None
for root, dirs, files in os.walk(BASE_DIR):
    if "RawNet3.py" in files:
        rawnet3_script_path = os.path.join(root, "RawNet3.py")
        break

if not rawnet3_script_path:
    # Fallback to look one directory up if needed
    for root, dirs, files in os.walk(os.path.dirname(BASE_DIR)):
        if "RawNet3.py" in files:
            rawnet3_script_path = os.path.join(root, "RawNet3.py")
            break

if not rawnet3_script_path:
    raise FileNotFoundError("Could not locate 'RawNet3.py' in project tree.")

parent_dir = os.path.dirname(os.path.abspath(rawnet3_script_path))
grandparent_dir = os.path.dirname(parent_dir)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

import RawNetBasicBlock
spec = importlib.util.spec_from_file_location("RawNet3_module", rawnet3_script_path)
rawnet3_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rawnet3_module)
RawNet3 = rawnet3_module.RawNet3

# 3. Initialize Model Once (Persistent in Memory)
model_kwargs = {
    "nOut": 256,
    "encoder_type": "ECA",
    "sinc_stride": 10,
    "out_bn": False,
    "log_sinc": True,
    "norm_sinc": "all"
}

model = RawNet3(
    RawNetBasicBlock.Bottle2neck,
    8,
    True,
    True,
    **model_kwargs
)

if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    if isinstance(state_dict, dict):
        state_dict = state_dict.get('state_dict', state_dict.get('model', state_dict))
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
model.eval()

# 4. Audio Preprocessor
def load_and_preprocess_audio(file_path: str, target_len: int = 48000) -> torch.Tensor:
    audio, sr = librosa.load(file_path, sr=16000, mono=True)
    
    # RMS Energy Normalization
    rms = np.sqrt(np.mean(audio**2)) + 1e-8
    audio = audio / rms * 0.1

    if len(audio) < target_len:
        repeats = (target_len // len(audio)) + 1
        audio = np.tile(audio, repeats)[:target_len]
    else:
        audio = audio[:target_len]

    return torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

# 5. Exported Inference Function
def evaluate_audio_authenticity(file_path: str) -> float:
    """
    Takes a path to an audio file and returns the deepfake spoof confidence (0.0 to 1.0).
    """
    tensor_audio = load_and_preprocess_audio(file_path)

    with torch.no_grad():
        embedding = model(tensor_audio)
        norm_emb = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        raw_var = torch.var(norm_emb).item()
        
        # Saksham's calculation logic
        prob = 1.0 / (1.0 + np.exp(-(raw_var - 0.0038) * 4000.0))
        real_percentage = float(np.clip(prob * 100.0, 1.0, 99.0))
        spoof_probability = (100.0 - real_percentage) / 100.0

    return spoof_probability
