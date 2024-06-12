import torch
from torch import nn
from transformers import WhisperProcessor, WhisperModel

class WhisperClassifier(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small.en")
        self.encoder = WhisperModel.from_pretrained("openai/whisper-small.en")

        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.encoder.encoder.parameters():
            param.requires_grad = True

        self.intent_classifier = nn.Sequential(
            nn.Linear(768, 14)
        )

    def forward(self, x):
        x = self.processor(x, sampling_rate=16000, return_tensors="pt")["input_values"].squeeze(0).to("cuda")
        x = self.encoder(x)
        x = torch.mean(x, dim=1)
        logits = self.intent_classifier(x)
        x.to("cpu")
        return logits