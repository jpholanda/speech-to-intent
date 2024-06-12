from models.ast import ASTClassifier
from models.hubert import HubertSSLClassifier
from models.wav2vec2 import Wav2VecClassifier
from models.whisper import WhisperClassifier
from trainer import LightningModel
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='push to hub')
    parser.add_argument('--model', dest='model', choices=['hubert', 'wav2vec2', 'whisper', 'ast'],
                        help='model to upload', required=True)
    parser.add_argument('--checkpoint', dest='checkpoint', help='checkpoint to load from', required=True)
    args = parser.parse_args()

    if args.model == "hubert":
        classifier = HubertSSLClassifier()
    elif args.model == "wav2vec2":
        classifier = Wav2VecClassifier()
    elif args.model == "ast":
        classifier = ASTClassifier()
    else:
        classifier = WhisperClassifier()

    # change path to checkpoint
    model = LightningModel.load_from_checkpoint(args.checkpoint, model=classifier)
    model.push_to_hub(f"speech-to-intent_{args.model}")