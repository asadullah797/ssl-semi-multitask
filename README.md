---
license: mit
pipeline_tag: audio-classification
tags:
- automatic-speech-recognition
- emotion-recognition
- speaker-identification
---

# Multitask Speech Model with Wav2Vec2

This repository contains a multitask learning pipeline built on top of Wav2Vec2
, designed to jointly perform:

Automatic Speech Recognition (ASR) (character-level CTC loss)

Speaker Identification

Emotion Recognition

The system is trained on a combination of training dataset with parallel data from speech transcriptions, speaker identification and emotion recognition labels.

📌 Features

Multitask model (Wav2Vec2MultiTasks) with shared Wav2Vec2 encoder and separate heads for:

Speech Recognition (CTC)

Speaker classification

Emotion classification

Custom data preprocessing:

Cleans transcripts (removes punctuation & special characters)

Converts numbers into words

Builds a vocabulary and tokenizer

Filters short/invalid audio

Training, validation, and test splits with collators for CTC.

Evaluation metrics:

Character Error Rate (CER) for character recognition

Accuracy for speaker and emotion classification
