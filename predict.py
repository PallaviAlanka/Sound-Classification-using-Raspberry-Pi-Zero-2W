from flask import Flask, jsonify, render_template
import pyaudio
import numpy as np
import librosa
import time
import threading
import tflite_runtime.interpreter as tflite

# Flask app
app = Flask(__name__)

# ==== Audio Settings ====
CHUNK = 2048
RATE = 16000
RECORD_SECONDS = 5
MFCC_COEFFS = 13
FORMAT = pyaudio.paInt16
CHANNELS = 1

# ==== AGC Settings ====
TARGET_RMS = 0.05  # Adjust target level as needed
GAIN_MAX = 10.0    # Optional safety cap for gain

# ==== Model Path ====
MODEL_PATH = 'model_rep.tflite'

# ==== Class Labels ====
labels = [
    "walker",          # index 0
    "dog barking",     # index 1
    "bird chirping",    # index 2  human talking
    "human talking",   # index 3
    "bathroom flush",  # index 4
    "basin tap"        # index 5
]

# Load model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Audio Stream
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)

# Shared data
latest_output = {"mfcc_shape": None, "output": None, "label": None, "confidence": None}

def audio_loop():
    global latest_output
    while True:
        frames = []
        print("⏳ Recording 5 seconds of audio...")
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            except IOError as e:
                print(f"⚠️ Buffer overflow: {e}")
                frames.append(b'\x00' * CHUNK * 2)

        # Convert audio bytes to float32 NumPy array
        audio_data = b''.join(frames)
        np_audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # === AGC Step ===
        rms = np.sqrt(np.mean(np_audio**2))
        if rms > 0:
            gain = min(TARGET_RMS / rms, GAIN_MAX)
            np_audio *= gain
            print(f"🔊 AGC applied. Gain: {gain:.2f}, Original RMS: {rms:.4f}")
        else:
            print("⚠️ Silence detected. Skipping gain adjustment.")

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=np_audio, sr=RATE, n_mfcc=MFCC_COEFFS)

        # Reshape/pad MFCCs to fit model input shape
        required_shape = input_details[0]['shape']  # e.g., (1, 13, 44, 1)
        mfccs = mfccs[:, :required_shape[2]]  # Trim if needed
        if mfccs.shape[1] < required_shape[2]:
            pad_width = required_shape[2] - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')

        input_tensor = np.expand_dims(mfccs, axis=0)  # Add batch dim
        if len(required_shape) == 4:
            input_tensor = np.expand_dims(input_tensor, axis=-1)  # Add channel dim

        input_tensor = input_tensor.astype(np.float32)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Get prediction
        probabilities = output_data[0]
        predicted_index = np.argmax(probabilities)
        predicted_label = labels[predicted_index]
        confidence = probabilities[predicted_index]

        # Convert NumPy types to native Python types (e.g., float32 to float)
        latest_output["mfcc_shape"] = str(mfccs.shape)
        latest_output["output"] = output_data.tolist()  # Convert to list for JSON serialization
        latest_output["label"] = predicted_label
        latest_output["confidence"] = float(confidence)  # Convert numpy.float32 to Python float

        # Print result
        print(f"✅ MFCCs shape: {mfccs.shape}")
        print(f"📊 Predicted Label: {predicted_label}")
        print(f"🔢 Confidence: {confidence:.4f}")
        print("-" * 40)

        time.sleep(0.1)

# Start audio processing thread
threading.Thread(target=audio_loop, daemon=True).start()

# === Flask Routes ===
@app.route("/")
def index():
    return render_template("index_1.html")

@app.route("/data")
def data():
    return jsonify(latest_output)

if __name__ == "__main__":
    app.run(host='192.168.221.152', port=5000)
