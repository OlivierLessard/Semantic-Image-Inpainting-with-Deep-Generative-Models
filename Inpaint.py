import argparse
from load_data import get_dataloaders, get_data


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gan-path", type=str, default="./checkpoints/model.pth")
    parser.add_argument("--eval-only", action='store_true', default=False)
    parser.add_argument("--test-only", action='store_true', default=False)

    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--optim-steps", type=int, default=1500)
    parser.add_argument("--blending-steps", type=int, default=3000)
    parser.add_argument("--prior-weight", type=float, default=0.003)
    parser.add_argument("--window-size", type=int, default=25)
    args = parser.parse_args()
    return args


def inpaint(args):
    svhn_train_data, svhn_test_data = get_data()
    svhn_train_data_loader, svhn_test_data_loader = get_dataloaders(svhn_train_data, svhn_test_data, args.batch_size)




if __name__ == "__main__":
    args = get_arguments()
    inpaint(args)