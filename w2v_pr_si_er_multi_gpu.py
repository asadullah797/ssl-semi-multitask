
import os
import re
import json
import torch
import evaluate
import torchaudio
import Levenshtein
import numpy as np
import pandas as pd
import torch.nn as nn
import soundfile as sf
from typing import Tuple
from tqdm.auto import tqdm
from torch.nn import CTCLoss
from num2words import num2words
import torch.nn.functional as F
from transformers import HubertModel
from transformers import Wav2Vec2Model
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from sklearn.metrics import accuracy_score
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from tokenizers.processors import TemplateProcessing
from datasets import load_dataset, Audio, Dataset
from typing import Any, Dict, List, Optional, Union

def setup_device_and_model(model):
    """
    Move model to CUDA if available and wrap with DataParallel if >1 GPU.
    Returns (model, device).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # Only wrap if not already wrapped
    if hasattr(torch.cuda, "device_count") and torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    model = model.to(device)
    return model, device

def save_checkpoint(model, path):
    """
    Save state_dict in a way that works for single-GPU and DataParallel.
    """
    state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state, path)

def load_checkpoint(model, path, map_location=None, strict=True):
    """
    Load state_dict handling DataParallel/non-DataParallel mismatches.
    """
    if map_location is None:
        map_location = "cpu"
    state = torch.load(path, map_location=map_location)
    try:
        model.load_state_dict(state, strict=strict)
    except Exception:
        # Try loading into .module for DataParallel models
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state, strict=strict)
        else:
            # Try removing potential 'module.' prefixes
            new_state = {}
            for k, v in state.items():
                new_state[k.replace("module.", "")] = v
            model.load_state_dict(new_state, strict=strict)
    return model
# === [END INSERTED BLOCK] ===


"""# Dataset"""


"""## Let's load dataset"""

timit_path = 'mix-dataset'
data_path = 'mix-dataset'
#csv_file = 'mix-dataset.csv'
csv_file = 'mix-dataset-filter.csv'

#def prepare_exp_dataset():
print('prepare_exp_dataset...')
df = pd.read_csv(os.path.join(data_path, csv_file))
data = {}
for idx, row in tqdm(df.iterrows()):
    row_list = row.tolist()
    data[idx] = row_list
    #if idx == 17430:
    #    break

def convert_to_feature_dict(data_dict):
    # convert each feature into an array instead
    audio_files = []
    word_files = []
    emotion_ids = []
    speaker_ids = []

    for key, value in data_dict.items():
        # audio_file = value[0].split('./')[1].replace('\\', '/')

        audio_file  = value[0].replace("\\", "/") # os.path.normpath(value[0])  # system format
        audio_files.append(os.path.join(timit_path, audio_file))
        speaker_ids.append(value[1])
        word_files.append(value[2])
        emotion_ids.append(value[3])

    return {
        'audio_file': audio_files,
        'speaker_id': speaker_ids,
        'word_file': word_files,
        'emotion_id': emotion_ids,

    }

raw_dataset = convert_to_feature_dict(data)
raw_dataset = Dataset.from_dict(raw_dataset)

def prepare_text_data(item):
    item['text'] = item['word_file']
    return item

raw_dataset = (raw_dataset
                .map(prepare_text_data)
                .remove_columns(["word_file",]))

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]$%&(*/\x85\x91\x92\x93\x94\x96\x97\xa0éó—’…'
pattern = r"[\[\]\\\?\.\!\-\;\:\"$%&\(\)\*/\x85\x91\x92\x93\x94\x96\x97\n\ub633\u7aca\uc9ed\uc9d9\xa0éó—’…]"

def remove_special_characters(batch):
    # batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower() + " "
    batch["text"] = re.sub(pattern, "", batch["text"].lower())
    return batch

raw_dataset = raw_dataset.map(remove_special_characters)

def is_valid_audio(path):
    try:
        data, sr = sf.read(path)
        return data.ndim > 0 and data.size > 0
    except:
        return False

raw_dataset = raw_dataset.filter(lambda x: is_valid_audio(x["audio_file"]))


raw_dataset = (raw_dataset
                .cast_column("audio_file", Audio(sampling_rate=16_000))
                .rename_column('audio_file', 'audio'))

MIN_DURATION = 2.0  # keep only audios longer than 2 seconds

def is_long_enough(example):
    audio = example["audio"]  # this is a dict with 'array' and 'sampling_rate'
    duration = len(audio["array"]) / audio["sampling_rate"]
    return duration >= MIN_DURATION

# Filter dataset
filtered_dataset = raw_dataset.filter(is_long_enough)

print(f"Original size: {len(raw_dataset)}")
print(f"Filtered size: {len(filtered_dataset)}")

def replace_numbers_with_words(text):
    return re.sub(r'\d+', lambda x: num2words(int(x.group())), text)

def norm_text(item):
    item['text'] = replace_numbers_with_words(item["text"])
    return item

filtered_dataset = filtered_dataset.map(norm_text)

def extract_vocab(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": vocab}
vocab = set()
for example in (filtered_dataset):
    vocab.update(list(example["text"].lower()))
vocab_dict = {v: i for i, v in enumerate(sorted(vocab))}

# make the space more intuitive to understand
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

print(len(vocab_dict))


# save vocab.json
with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(".", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|", )  # './' load vocab.json in the current directory
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# Map speaker IDs to integer labels
speaker_ids = list(set(filtered_dataset['speaker_id']))
speaker_to_id = {spk: i for i, spk in enumerate(sorted(speaker_ids))}
print(len(speaker_to_id))

# Map speaker IDs to integer labels
emotion_id = list(set(filtered_dataset['emotion_id']))
emotion_to_id = {spk: i for i, spk in enumerate(sorted(emotion_id))}
print(len(emotion_to_id))

def resample_audio(audio_array, orig_sr, target_sr=16000):
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    audio_tensor = torch.tensor(audio_array, dtype=torch.float32)  # ✅ use float32
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)  # make it (1, n) shape
    resampled = resampler(audio_tensor)
    return resampled.squeeze().numpy()  # remove channel dim again

def prepare_dataset(batch):
    audio = batch["audio"]

    if audio["sampling_rate"] != 16000:
        audio_array = resample_audio(audio["array"], audio["sampling_rate"], 16000)
    else:
        audio_array = audio["array"]

    batch["input_values"] = processor(audio_array, sampling_rate=16000).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
        batch["emotion_labels"] = emotion_to_id[batch["emotion_id"]]
        batch["speaker_labels"] = speaker_to_id[batch["speaker_id"]]
    return batch

filtered_dataset = filtered_dataset.map(prepare_dataset)

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt",)
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=self.padding, return_tensors="pt",)
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        if "speaker_labels" in features[0]:
            batch["speaker_labels"] = torch.tensor([feature["speaker_labels"] for feature in features], dtype=torch.long)
        if "emotion_labels" in features[0]:
            batch["emotion_labels"] = torch.tensor([feature["emotion_labels"] for feature in features], dtype=torch.long)

        return batch

split_dataset = filtered_dataset.train_test_split(test_size=0.6, seed=42)
train_dataset = split_dataset['train']
split_dataset = split_dataset['test'].train_test_split(test_size=0.5, seed=42)
valid_dataset = split_dataset['train']
test_dataset = split_dataset['train']


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)


train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=data_collator)
valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=True, collate_fn=data_collator)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
# return train_dataloader, valid_dataloader, test_dataset, tokenizer, vocab_dict, speaker_to_id, emotion_to_id


class Wav2Vec2MultiTasks(nn.Module):
    def __init__(self, base_model="facebook/wav2vec2-base", num_phonemes=len(vocab_dict), num_speakers=len(speaker_to_id), num_emotions=len(emotion_to_id)):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(base_model)
        self.dropout = nn.Dropout(0.1)
        self.phoneme_classifier = nn.Linear(self.wav2vec.config.hidden_size, num_phonemes)
        self.speaker_classifier = nn.Linear(self.wav2vec.config.hidden_size, num_speakers)
        self.speaker_classifier = nn.Linear(self.wav2vec.config.hidden_size, num_speakers)
        self.emotion_classifier = nn.Linear(self.wav2vec.config.hidden_size, num_emotions)

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, T, H)
        phoneme_logits = self.phoneme_classifier(self.dropout(hidden_states))
        pooled = hidden_states.mean(dim=1)
        speaker_logits = self.speaker_classifier(self.dropout(pooled))
        emotion_logits = self.emotion_classifier(self.dropout(pooled))

        return phoneme_logits, speaker_logits, emotion_logits
def multitask_loss(phoneme_logits, phoneme_labels, speaker_logits, speaker_labels, emotion_logits, emotion_labels):
    ce_loss_fn = nn.CrossEntropyLoss()
    speaker_loss = ce_loss_fn(speaker_logits, speaker_labels)
    emotion_loss = ce_loss_fn(emotion_logits, emotion_labels)

    # --- Phoneme Loss (CTC) ---
    ctc_loss_fn = nn.CTCLoss(blank=processor.tokenizer.pad_token_id, reduction="mean", zero_infinity=True)
    log_probs = F.log_softmax(phoneme_logits, dim=-1).transpose(0, 1)  # [T, B, V]
    batch_size = phoneme_logits.size(0)
    input_lengths = torch.full(size=(batch_size,), fill_value=phoneme_logits.size(1), dtype=torch.long).to(phoneme_logits.device)
    if isinstance(phoneme_labels, torch.Tensor):
        phoneme_labels_list = [x[x != -100] for x in phoneme_labels]  # Remove padding tokens
    else:
        phoneme_labels_list = phoneme_labels
    targets = torch.cat(phoneme_labels_list).to(torch.long).to(phoneme_logits.device)
    target_lengths = torch.tensor([len(t) for t in phoneme_labels_list], dtype=torch.long).to(phoneme_logits.device)

    assert all(input_lengths[i] >= target_lengths[i] for i in range(batch_size)), \
        f"CTC Error: input_lengths must be >= target_lengths. Got: {input_lengths} vs {target_lengths}"
    phoneme_loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
    total_loss = 0.65 * phoneme_loss + 0.2 * speaker_loss + 0.15 * emotion_loss

    return total_loss, phoneme_loss, speaker_loss, emotion_loss

"""## Training"""
def start_training(train_dataloader, valid_dataloader, vocab_dict, speaker_to_id, emotion_to_id):

    # model = Wav2Vec2MultiTask(num_phonemes=len(vocab_dict), num_speakers=len(speaker_to_id), num_emotions=len(emotion_to_id))
    model = Wav2Vec2MultiTasks(num_phonemes=len(vocab_dict), num_speakers=len(speaker_to_id), num_emotions=len(emotion_to_id))
    model, device = setup_device_and_model(model)  # inserted for multi-GPU
    model = load_checkpoint(model, "w2v_model.pth", device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    num_epochs = 5
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:

            input_values   = torch.tensor(batch["input_values"]).to(device)
            phoneme_labels = torch.tensor(batch["labels"]).to(device)
            speaker_labels = torch.tensor(batch["speaker_labels"]).to(device)
            emotion_labels = torch.tensor(batch["emotion_labels"]).to(device)
            phoneme_logits, speaker_logits, emotion_logits = model(input_values)
            loss, phoneme_loss, speaker_loss, emotion_loss  = multitask_loss(phoneme_logits, phoneme_labels, speaker_logits, speaker_labels, emotion_logits, emotion_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}, phoneme_loss: {phoneme_loss.item():.4f}, speaker_loss: {speaker_loss.item():.4f}, emotion_loss: {emotion_loss.item():.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in valid_dataloader:
                input_values   = torch.tensor(batch["input_values"]).to(device)
                phoneme_labels = torch.tensor(batch["labels"]).to(device)
                speaker_labels = torch.tensor(batch["speaker_labels"]).to(device)
                emotion_labels = torch.tensor(batch["emotion_labels"]).to(device)
                phoneme_logits, speaker_logits, emotion_logits = model(input_values)
                loss, phoneme_loss, speaker_loss, emotion_loss  = multitask_loss(phoneme_logits, phoneme_labels, speaker_logits, speaker_labels, emotion_logits, emotion_labels)
                val_loss += loss.item()
        val_loss /= len(valid_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} | Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, "w2v_model.pth")
            print(f"✅ Saved new best model at epoch {epoch+1} with val_loss={val_loss:.4f}")


def model_evaluation(test_dataset, tokenizer, vocab_dict, speaker_to_id, emotion_to_id):

    # Load best model checkpoint;
    model = Wav2Vec2MultiTasks(num_phonemes=len(vocab_dict), num_speakers=len(speaker_to_id), num_emotions=len(emotion_to_id))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(model, "w2v_model.pth", device)
    model = model.to(device)

    PAD_ID = tokenizer.encode("[PAD]")[0]
    EMPTY_ID = tokenizer.encode(" ")[0]

    def collapse_tokens(tokens: List[Union[str, int]]) -> List[Union[str, int]]:
        prev_token = None
        out = []
        for token in tokens:
            if token != prev_token and prev_token is not None:
                out.append(prev_token)
            prev_token = token
        return out

    def clean_token_ids(token_ids: List[int]) -> List[int]:
        """
        Remove [PAD] and collapse duplicated token_ids
        """
        token_ids = [x for x in token_ids if x not in [PAD_ID, EMPTY_ID]]
        token_ids = collapse_tokens(token_ids)
        return token_ids

    cer_metric = evaluate.load("cer")

    def safe_cer(cer_metric, predictions, references):
        total_chars = sum(len(r) for r in references)
        if total_chars <= 2:
            return 0.0
        return cer_metric.compute(predictions=predictions, references=references)

    result = []
    spk_preds = []
    spk_labels = []
    emotion_preds = []
    emotion_labels = []

    for x in test_dataset:
        label_ids = torch.tensor(x["labels"]).unsqueeze(0).to(device)
        input_values = torch.tensor(x["input_values"]).unsqueeze(0).to(device)
        speaker_label = x["speaker_labels"]
        emotion_label = x["emotion_labels"]

        with torch.no_grad():
            phoneme_logits, speaker_logits, emotion_logits = model(input_values)
            predicted_ids = torch.argmax(phoneme_logits, dim=-1)
            speaker_pred = torch.argmax(speaker_logits, dim=-1).item()
            emotion_pred = torch.argmax(emotion_logits, dim=-1).item()

            # Convert ID to Char because Levenstein library operates on char level
            predicted_ids = clean_token_ids(predicted_ids[0].tolist())
            predicted_str = tokenizer.decode(predicted_ids, group_tokens=False)
            predicted_chr = "".join([chr(x) for x in predicted_ids])

            label_ids = clean_token_ids(label_ids[0].int().tolist())
            label_str = tokenizer.decode(label_ids, group_tokens=False)
            label_chr = "".join([chr(x) for x in label_ids])

            # Compute CER (character-error rate). Each character is IPA character
            cer_score = safe_cer(cer_metric, [predicted_chr], [label_chr])
            result.append((predicted_str, label_str, cer_score))

            spk_labels.append(speaker_label)
            spk_preds.append(speaker_pred)

            emotion_preds.append(emotion_pred)
            emotion_labels.append(emotion_label)

    print(predicted_str)

    print(label_str)

    print(x['text'])

    """# Phoneme Recognition task"""

    # Recalculate CER store for Test Set
    total_cer = 0
    for entry in result:
        total_cer += entry[-1]
    print(f"Character accuracy: {1.0 - total_cer / len(result)}")

    """# Speaker Identification task"""

    acc = accuracy_score(spk_labels, spk_preds)
    print('speaker accuracy', acc)

    """# Emotion Recognition
    """

    acc = accuracy_score(emotion_labels, emotion_preds)
    print('emotion accuracy', acc)

def main():
    
    # train_dataloader, valid_dataloader, test_dataset, tokenizer, vocab_dict, speaker_to_id, emotion_to_id = prepare_exp_dataset()
    start_training(train_dataloader, valid_dataloader, vocab_dict, speaker_to_id, emotion_to_id)
    model_evaluation(test_dataset, tokenizer, vocab_dict, speaker_to_id, emotion_to_id)

if __name__ == "__main__":
    main()
