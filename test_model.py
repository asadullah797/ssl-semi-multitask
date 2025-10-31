import json
import torch
import soundfile as sf
from transformers import Wav2Vec2Config
from modeling_wav2vec2_multitask import Wav2Vec2MultiTask
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from configuration_wav2vec2_multitask import Wav2Vec2MultiTaskConfig

base_config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base")
multi_task_config = Wav2Vec2MultiTaskConfig(
    **base_config.to_dict(),
    num_emotions=14,
    num_phonemes=33,
    num_speakers=373
)
# Step 1: Create config
config = Wav2Vec2MultiTaskConfig.from_pretrained("config.json")

model = Wav2Vec2MultiTask(multi_task_config)

speech, sample_rate = sf.read("test.wav")
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
processor = Wav2Vec2Processor.from_pretrained(".")   # base vocab

inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
input_values = inputs.input_values  # this is what goes to model

phoneme_logits, speaker_logits, emotion_logits = model(input_values)

print("Phoneme logits:", phoneme_logits.shape)
print("Speaker logits:", speaker_logits.shape)
print("Emotion logits:", emotion_logits.shape)
speaker_pred = torch.argmax(speaker_logits, dim=-1)
emotion_pred = torch.argmax(emotion_logits, dim=-1)
with open("emotion_map.json", "r") as f:
    emotion_to_id = json.load(f)

emotion_to_id = {v: k for k, v in emotion_to_id.items()}
print("Emotion label:", emotion_to_id[emotion_pred.item()])

with open("speaker_map.json", "r") as f:
    speaker_to_id = json.load(f)

speaker_to_id = {v: k for k, v in speaker_to_id.items()}
print("Emotion label:", speaker_to_id[speaker_pred.item()])

predicted_ids = torch.argmax(phoneme_logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)[0]
transcription = transcription.replace("[UNK]", "").replace("<s>", "").replace("</s>", "").replace("<pad>", "")

print("Transcription:", transcription)
