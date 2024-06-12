import torch
from torch import nn
from transformers import Wav2Vec2Processor, HubertModel


class HubertSSLClassifier(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.encoder = HubertModel.from_pretrained("facebook/hubert-large-ll60k")

        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.encoder.encoder.parameters():
            param.requires_grad = True

        self.intent_classifier = nn.Sequential(
            nn.Linear(1024, 14),
        )

    def forward(self, x):
        x = self.processor(x, sampling_rate=16000, return_tensors="pt")["input_values"].squeeze(0).to("cuda")
        x = self.encoder(x).last_hidden_state
        x = torch.mean(x, dim=1)
        logits = self.intent_classifier(x)
        x.to("cpu")
        return logits