from flask import Flask, request, jsonify
import base64
from transformers import pipeline
import numpy as np
import matplotlib.pyplot as plt
from transformers import WhisperFeatureExtractor, WhisperTokenizer

app = Flask(__name__)

MODEL_NAME = 'biodatlab/whisper-th-medium-combined'
PATH = './model.pth'

model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.generation_config.language = "Thai"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

device = 0 if torch.cuda.is_available() else "cpu"

# feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
# processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="Thai", task="transcribe")

# def prepare_dataset(batch):
    
#     audio = batch["audio"]
#     batch["input_features"] = feature_extractor(audio["array"], 
#                               sampling_rate=audio["sampling_rate"]).input_features[0]
#     del batch["audio"]
#     return batch

# @dataclass
# class DataCollatorSpeechSeq2SeqWithPadding:
#     processor: Any

#     def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
#         input_features = [{"input_features": feature["input_features"]} for feature in features]
#         batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

#         return batch

# data_collator = DataCollatorSpeechSeq2SeqWithPadding(
#     processor=processor,
# )

pipeline = pipeline(
    task="automatic-speech-recognition",
    model=model,
    chunk_length_s=30,
    device=device,
)

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    # Ensure the request is a POST with JSON data
    if request.method == 'POST' and request.is_json:

        audio_np, sample_rate = data_collator(prepare_dataset(request.json))

        try:
            transcription = transcribe_audio_array(audio_np, sample_rate)
            return jsonify({'transcription': transcription}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Expected POST request with JSON data'}), 400

def transcribe_audio_array(audio_np, sample_rate):
    
    # pipe(request.json, generate_kwargs={"language":"<|th|>", "task":"transcribe"}, batch_size=16)["text"]
    transcription = pipeline(audio_np, sampling_rate=sample_rate)[0]['transcription']
    return transcription

if __name__ == '__main__':
    app.run(debug=True)