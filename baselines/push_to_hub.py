from hubert.trainer import LightningModel as HubertModel
from wav2vec2.trainer import LightningModel as Wav2VecModel
from whisper.trainer import LightningModel as WhisperModel
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='push to hub')
    parser.add_argument('--model', dest='model', choices=['hubert', 'wav2vec2', 'whisper'],
                        help='model to upload', required=True)
    parser.add_argument('--checkpoint', dest='checkpoint', help='checkpoint to load from', required=True)
    args = parser.parse_args()

    if args.model == 'hubert':
        model = HubertModel.load_from_checkpoint(args.checkpoint)
    elif args.model == 'wav2vec2':
        model = Wav2VecModel.load_from_checkpoint(args.checkpoint)
    else:
        model = WhisperModel.load_from_checkpoint(args.checkpoint)

    model.push_to_hub(f"speech-to-intent_{args.model}")