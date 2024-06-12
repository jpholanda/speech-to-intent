import argparse

NUM_WORKERS=16
BATCH_SIZE=2

def get_arguments():
    parser = argparse.ArgumentParser(description='model trainer')

    parser.add_argument('--model', choices=["hubert", "wav2vec2", "whisper", "ast"], required=True)

    parser.add_argument('--dev',
                        dest='dev',
                        action='store_true',
                        help='enables dev mode (runs only one step)')
    parser.add_argument('--prod',
                        dest='dev',
                        action='store_false',
                        help='do not enable dev mode (runs normally)')
    parser.set_defaults(dev=False)

    parser.add_argument('--checkpoint', dest='checkpoint', help='checkpoint to resume training from')

    args = parser.parse_args()
    return args