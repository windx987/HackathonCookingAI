from flask import Flask, request, jsonify
from transformers import (WhisperFeatureExtractor, 
                          WhisperTokenizer, 
                          WhisperProcessor, 
                          WhisperForConditionalGeneration)

app = Flask(__name__)

MODEL_NAME = 'biodatlab/whisper-th-medium-combined'

# Initialize processor before it's used
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="Thai", task="transcribe")

model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.generation_config.language = "Thai"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

model.to(device)

@app.route('/transcribe', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400
    
    audio_file = request.files['audio']
    audio_data = audio_file.read()

    # Perform ASR
    transcription = model.transcribe(audio_data)
    
    return jsonify({'transcription': transcription})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

