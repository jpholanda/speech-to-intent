import torch
from torch import nn
from transformers import ASTFeatureExtractor, ASTModel

class ASTClassifier(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.processor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.encoder = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.encoder.encoder.parameters():
            param.requires_grad = True

        self.intent_classifier = nn.Sequential(
            nn.Linear(768, 14),
        )

    def forward(self, x):
        x = self.processor(x.to("cpu"), sampling_rate=16000, return_tensors="pt")["input_values"].squeeze(0).to("cuda")
        x = self.encoder(x).last_hidden_state
        x = torch.mean(x, dim=1)
        logits = self.intent_classifier(x)
        x.to("cpu")
        return logits