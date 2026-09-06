import os
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly

folder = "test_audio"
target_file = None

for file in os.listdir(folder):
    if file.endswith((".ogg", ".m4a", ".opus")) or file.startswith("WhatsApp"):
        target_file = os.path.join(folder, file)
        break

if target_file:
    print(f"Loading: {target_file}")
    output_file = os.path.join(folder, "real.wav")
    
    # Direct read via soundfile
    data, orig_sr = sf.read(target_file)
    
    # Stereo to Mono conversion
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
        
    # Resample to 16000 Hz if needed
    if orig_sr != 16000:
        print(f"Resampling from {orig_sr} Hz to 16000 Hz...")
        data = resample_poly(data, 16000, orig_sr)
        
    # Write normalized 16kHz WAV
    sf.write(output_file, data.astype(np.float32), 16000)
    print(f"✅ Converted and Resampled cleanly to 16000 Hz -> {output_file}")
else:
    print(f"❌ Target file not found in {folder}/")