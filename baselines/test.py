import argparse

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)

import sys
sys.path.append("/root/Speech2Intent/s2i-baselines")

import torch.nn.functional as F

# choose the model
from trainer_hubert import LightningModel
# from trainer_whisper import LightningModel
# from trainer_wav2vec2 import LightningModel

from dataset import S2IDataset

from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='push to hub')
    parser.add_argument('--checkpoint', dest='checkpoint', help='checkpoint to load from', required=True)
    args = parser.parse_args()

    dataset = S2IDataset(split='test')

    # change path to checkpoint
    model = LightningModel.load_from_checkpoint(args.checkpoint)
    model.to('cuda')
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