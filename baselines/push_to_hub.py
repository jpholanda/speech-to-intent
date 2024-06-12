from trainer import LightningModel
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='push to hub')
    parser.add_argument('--model', dest='model', choices=['hubert', 'wav2vec2', 'whisper'],
                        help='model to upload', required=True)
    parser.add_argument('--checkpoint', dest='checkpoint', help='checkpoint to load from', required=True)
    args = parser.parse_args()

    model = LightningModel.load_from_checkpoint(args.checkpoint)
    model.push_to_hub(f"speech-to-intent_{args.model}")