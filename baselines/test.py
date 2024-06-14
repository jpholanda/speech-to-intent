import argparse

import warnings

from models.ast import ASTClassifier
from models.hubert import HubertSSLClassifier
from models.wav2vec2 import Wav2VecClassifier
from models.whisper import WhisperClassifier

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)

import sys
sys.path.append("/root/Speech2Intent/s2i-baselines")

import torch.nn.functional as F
from trainer import LightningModel

from dataset import S2IDataset

from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test model')
    parser.add_argument('--model', dest='model', choices=['hubert', 'wav2vec2', 'whisper', 'ast'],
                        help='model to test', required=True)
    parser.add_argument('--checkpoint', dest='checkpoint', help='checkpoint to load from', required=True)
    parser.add_argument('--device', dest='device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()

    dataset = S2IDataset(split='test')

    if args.model == "hubert":
        classifier = HubertSSLClassifier()
    elif args.model == "wav2vec2":
        classifier = Wav2VecClassifier(device=args.device)
    elif args.model == "ast":
        classifier = ASTClassifier()
    else:
        classifier = WhisperClassifier()

    # change path to checkpoint
    model = LightningModel.load_from_checkpoint(args.checkpoint, model=classifier)
    model.to(args.device)
    model.eval()

    trues=[]
    preds = []

    for x, label in tqdm(dataset):
        y_hat_l = model(x)

        probs = F.softmax(y_hat_l, dim=1).detach().cpu().view(1, 14)
        pred = probs.argmax(dim=1).detach().cpu().numpy().astype(int)
        probs = probs.numpy().astype(float).tolist()
        trues.append(label)
        preds.append(pred)

    print(f"Accuracy Score = {accuracy_score(trues, preds)}\nF1-Score = {f1_score(trues, preds, average='weighted')}")