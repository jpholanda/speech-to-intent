import argparse

NUM_WORKERS=1

def get_arguments():
    parser = argparse.ArgumentParser(description='model trainer')
    parser.add_argument('--dev',
                        dest='dev',
                        type=bool,
                        default=False,
                        action=argparse.BooleanOptionalAction,
                        help='enables dev mode (runs only one step to check if everything\'s working)')
    args = parser.parse_args()
    return args