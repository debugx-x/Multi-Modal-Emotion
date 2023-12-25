# Multi-Modal Sarcasm Detection using Multi-Task Learning

This repository covers the code for the Multi-Modal Sarcasm Detection project on the [Mustard](https://github.com/soujanyaporia/MUStARD) dataset using Multi-task learning.

# Data
Data can be downloaded over google drive with this link: https://drive.google.com/drive/folders/1R1MkRyejSe5IbUXZClUbQy1HKzR57Way?usp=sharing

## Instructions for use

# TODO Fill in instructions

first 

module load git-lfs
source venv/bin/activate
wandb sweep --project LSTM_text -e ddi --name hyper_param_tuning_lstm ../hyper_parameter_config/lstm.yaml
### Training

text_nn.py -lr 0.001 -e 3 -b 16 -w 0.000001 -s 32 -d "data/emotion_pd" -c 100 -p 10 -m LSTM -t 5 -y 7 -hl 300

text_nn.py -lr 0.001 -e 3 -b 16 -w 0.000001 -s 32 -d "data/emotion_pd" -c 100 -p 10 -m Bert -t 5 -y 7 -hl 300

audio_nn_wav2vec.py -d data/emotion_pd_raw.pkl -m Wav2Vec2 -y 7 -b 8  -o 1024
## Inference
