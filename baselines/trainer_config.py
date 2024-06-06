import argparse

NUM_WORKERS=1

def get_arguments():
    parser = argparse.ArgumentParser(description='model trainer')
    parser.add_argument('--dev',
                        dest='dev',
                        action='store_true',
                        help='enables dev mode (runs only one step)')
    parser.add_argument('--prod',
                        dest='dev',
                        action='store_false',
                        help='do not enable dev mode (runs normally)')
    parser.set_defaults(dev=False)

    args = parser.parse_args()
    return args