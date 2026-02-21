# Wav2Vec2-MultiTask

This is a fine-tuned **Wav2Vec2.0** model for **multi-task learning**:
- Phoneme recognition
- Emotion classification
- Speaker identification

## Usage

```python
from transformers import AutoModel, AutoConfig, AutoProcessor

model = AutoModel.from_pretrained(
    "asadullah797/my-wav2vec2-multitask",
    trust_remote_code=True
)

config = AutoConfig.from_pretrained(
    "username/my-wav2vec2-multitask",
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")

inputs = processor("hello world", return_tensors="pt", sampling_rate=16000)

# phoneme recognition
logits = model(**inputs, task="phoneme")
