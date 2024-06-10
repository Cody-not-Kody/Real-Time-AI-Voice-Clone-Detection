import pandas as pd
import numpy as np 
import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sys
import time
import matplotlib.pyplot as plt
import librosa
from IPython.display import Audio
from sklearn.metrics import roc_auc_score
from sklearn import metrics

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class Audio_Data(Dataset):

    def __init__(self, metadata, dir, transform, target_rate, num_samples, device):
        self.metadata = pd.read_csv(metadata)
        self.dir = dir
        self.device = device
        self.transform = transform.to(self.device)
        self.target_rate = target_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_path(index)
        label = self._get_audio_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transform(signal)
        return signal, label

    def _get_audio_path(self, index):
        filename = f"{self.metadata.iloc[index,0]}"
        audio_path = os.path.join(self.dir, filename)
        return audio_path

    def _get_audio_label(self, index):
        return self.metadata.iloc[index,1]

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_rate).cuda()
            signal = resampler(signal)
        return signal
    
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal 

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        signal_len = signal.shape[1]
        if signal_len < self.num_samples:
            difference = self.num_samples - signal_len
            last_dim_padding = (0, difference)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

class DNN(nn.Module):
    def __init__(self, num_fc_layers):
        super(DNN, self).__init__()
        self.num_fc_layers = num_fc_layers

        # INPUT SHAPE = (BATCH SIZE, NUM_CHANNEL, NUM_MELS, NUM_FEATS)
        # INPUT SHAPE = (BATCH SIZE, 1, 64, 219) WHEN SAMPLE RATE = 16000 AND DURATION = 7 SECS
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,         #Number of input channels; spectrograms will be treated as grayscale images
                out_channels = 32,       #Number of filters in convolutional layer
                kernel_size = 5,         
            ),
            nn.MaxPool2d(kernel_size = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),                  
            nn.Conv2d(
                in_channels = 32,        #Number of input channels from previous convolution
                out_channels = 64,       #Number of filters in convolutional layer
                kernel_size = 5,         
            ),
            nn.MaxPool2d(kernel_size = 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p = 0.25),
            nn.Flatten(start_dim = 2)
        )
        # OUTPUT SHAPE = (BATCH SIZE, OUT_CHANNELS OF LAST CONV, FLATTEN)

        self.gru  = nn.GRU(64, 128, num_layers = self.num_fc_layers, batch_first = True) # INPUT SIZE IS SAME AS NUMBER OF CHANNELS FROM LAST CNN LAYER
        
        self.fc_block = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,2)
        )
        
    def forward(self, input_data):         
        h0 = torch.zeros(self.num_fc_layers, input_data.shape[0], 128).to(device) # (num_layers, batch size, hidden size)
        x = self.conv_block(input_data)
        # print(x.shape)
        x = x.reshape(-1, x.shape[2], x.shape[1]) # SHAPE: (BATCH SIZE, SEQ LENGTH or NUM FEATS, INPUT SIZE or NUM ROWS)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        logits = F.sigmoid(self.fc_block(out))
        return logits


def create_data_loader(data, batch_size):
    dataloader = DataLoader(data, batch_size=batch_size)
    return dataloader   

def train(model, train_dataloader, val_dataloader, loss_fn, optimizer, device, epochs):           #Training the model
    model.train()
    train_loss = []
    val_loss = []
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        epoch_train_loss, epoch_val_loss = train_single_epoch(model, train_dataloader, val_dataloader, loss_fn, optimizer, device)
        train_loss.append(epoch_train_loss)
        val_loss.append(epoch_val_loss)
        print("--------------------------")
    print("Finished training")
    return train_loss, val_loss

def train_single_epoch(model, train_dataloader, val_dataloader, loss_fn, optimizer, device):
    epoch_train_loss = 0.0
    num_train_batches = 0

    epoch_val_loss = 0.0
    num_val_batches = 0

    for input, target in train_dataloader:
        input, target = input.to(device), target.to(device)

        #Calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        #Backpropagate error and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
        num_train_batches += 1
        epoch_train_loss = epoch_train_loss / num_train_batches

    print(f"Train loss: {epoch_train_loss:.4f}")

    with torch.no_grad():
        for input, target in val_dataloader:
            input, target = input.to(device), target.to(device)

            #Calculate loss
            prediction = model(input)
            loss = loss_fn(prediction, target)

            epoch_val_loss += loss.item()
            num_val_batches += 1
            epoch_val_loss = epoch_val_loss / num_val_batches
        
        print(f"Val loss: {epoch_val_loss:.4f}")

    return epoch_train_loss, epoch_val_loss



if __name__ == '__main__': 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Hyper-parameters for Melspectrogram transformation 
    sample_rate = 22050
    num_samples = 154350
    n_fft = 1024
    win_length = None
    hop_length = 512
    n_mels = 64

    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        n_mels=n_mels,
        mel_scale="htk",
    )

    Train_metadata_path = 'insert training metadata file path'
    Train_dir_path = 'insert training dataset path' 

    Valid_metadata_path = 'insert validation metadata file path'
    Valid_dir_path = 'insert validation dataset path' 

    batch_size = 128

    # Training audio data instance
    Train_audio_data = Audio_Data(Train_metadata_path, Train_dir_path, mel_spec, sample_rate, num_samples, device)
    print("Training set length:", f"{len(Train_audio_data)}")
    train_dataloader = create_data_loader(Train_audio_data, batch_size)

    # Validation audio data instance
    Valid_audio_data = Audio_Data(Valid_metadata_path, Valid_dir_path, mel_spec, sample_rate, num_samples, device)
    print("Validation set length:", f"{len(Valid_audio_data)}")
    val_dataloader = create_data_loader(Valid_audio_data, batch_size)

    loss_fn = nn.CrossEntropyLoss()   

    model = DNN(4).to(device)  # initializing model, parameter passed determines number of FC layers
    model_opt = optim.Adam(model.parameters(), lr = 0.0001)
    print(model)

    EPOCHS = 100

    train_loss, val_loss = train(model, train_dataloader, val_dataloader, loss_fn, model_opt, device, EPOCHS)  # Training the model

    model_name = 'name.pth'  # provide model name for each model trained
    torch.save(model.state_dict(), model_name)               
    print("Trained neural network saved at", model_name)
    