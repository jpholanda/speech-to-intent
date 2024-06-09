from trainer_config import get_arguments
from trainer_hubert import LightningModel as HubertModel
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='push to hub')
    parser.add_argument('--checkpoint', dest='checkpoint', help='checkpoint to load from', required=True)
    args = parser.parse_args()

    model = HubertModel.load_from_checkpoint(args.checkpoint)
    model.push_to_hub("speech-to-intent_hubert")