import os
import pandas as pd

import torch
import torchaudio
import torch.nn.utils.rnn as rnn_utils

from models import whisper

DATASET_ROOT = "/home/jpardholanda/speech-to-intent/dataset"

def collate_fn(batch):
    (seq, label) = zip(*batch)
    seql = [x.reshape(-1,) for x in seq]
    data = rnn_utils.pad_sequence(seql, batch_first=True, padding_value=0)
    label = torch.tensor(list(label))
    return data, label

def collate_mel_fn(batch):
    (seq, label) = zip(*batch)
    data = torch.stack([x.reshape(80, -1) for x in seq])
    label = torch.tensor(list(label))
    return data, label

class S2IDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_root=DATASET_ROOT, split='train'):
        csv_path = os.path.join(dataset_root, f'{split}.csv')
        wav_dir_path = dataset_root

        self.df = pd.read_csv(csv_path)
        self.wav_dir = wav_dir_path
        self.resmaple = torchaudio.transforms.Resample(8000, 16000)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        row = self.df.iloc[idx]
        intent_class = row["intent_class"]
        wav_path = os.path.join(self.wav_dir, row["audio_path"])
        speaker_id = row["speaker_id"]
        template = row["template"]

        wav_tensor, _= torchaudio.load(wav_path)
        wav_tensor = self.resmaple(wav_tensor)
        intent_class = int(intent_class)
        return wav_tensor, intent_class

class S2IMELDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_root=DATASET_ROOT, split='train'):
        csv_path = os.path.join(dataset_root, f'{split}.csv')
        wav_dir_path = dataset_root

        self.df = pd.read_csv(csv_path)
        self.wav_dir = wav_dir_path
        self.resmaple = torchaudio.transforms.Resample(8000, 16000)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        row = self.df.iloc[idx]
        intent_class = row["intent_class"]
        wav_path = os.path.join(self.wav_dir, row["audio_path"])
        speaker_id = row["speaker_id"]
        template = row["template"]

        wav_tensor, _= torchaudio.load(wav_path)
        wav_tensor = self.resmaple(wav_tensor)
            
        wav_tensor = whisper.pad_or_trim(wav_tensor.flatten())
        mel = whisper.log_mel_spectrogram(wav_tensor)

        intent_class = int(intent_class)
        return mel, intent_class

if __name__ == "__main__":
    dataset1 = S2IMELDataset() #,
    dataset2 = S2IDataset() #,
#        sr=16000)
    wav_tensor, intent_class = dataset1[0] 
    print(wav_tensor.shape, intent_class)

    wav_tensor, intent_class = dataset2[0] 
    print(wav_tensor.shape, intent_class)

    trainloader1 = torch.utils.data.DataLoader(
            dataset1, 
            batch_size=3, 
            shuffle=True, 
            num_workers=4,
            collate_fn = collate_mel_fn,
        )
    trainloader2 = torch.utils.data.DataLoader(
            dataset2, 
            batch_size=3, 
            shuffle=True, 
            num_workers=4,
            collate_fn = collate_fn,
        )
    x, y = next(iter(trainloader1))
    print(x.shape)
    print(y.shape)
    x, y = next(iter(trainloader2))
    print(x.shape)
    print(y.shape)
