# stdlib
import argparse
import os
# 3p
import numpy as np
import torch
# project
from model import FMNet
from faust_dataset import FAUSTDataset
from loss import SoftErrorLoss


def train_fmnet(args):
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # create dataset
    dataset = FAUSTDataset(args.dataroot)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.n_cpu)
    # create model
    fmnet = FMNet(n_residual_blocks=args.num_blocks, in_dim=352).to(device)  # number of features of SHOT descriptor
    optimizer = torch.optim.Adam(fmnet.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    criterion = SoftErrorLoss().to(device)

    # Training loop
    iterations = 0
    for epoch in range(1, args.n_epochs + 1):
        for i, data in enumerate(dataloader):
            data = [x.to(device) for x in data]
            feat_x, evecs_x, evecs_trans_x, dist_x, feat_y, evecs_y, evecs_trans_y, dist_y = data

            # sample vertices
            vertices = np.random.choice(feat_x.size(1), args.n_vertices)
            feat_x, evecs_x, evecs_trans_x = feat_x[:, vertices, :], evecs_x[:, vertices, :], evecs_trans_x[:, :, vertices]
            feat_y, evecs_y, evecs_trans_y = feat_y[:, vertices, :], evecs_y[:, vertices, :], evecs_trans_y[:, :, vertices]
            dist_x, dist_y = dist_x[:, vertices, :][:, :, vertices], dist_y[:, vertices, :][:, :, vertices]

            # do iteration
            fmnet.train()
            P, _ = fmnet(feat_x, feat_y, evecs_x, evecs_y, evecs_trans_x, evecs_trans_y)
            loss = criterion(P, dist_y)
            loss.backward()
            if (i + 1) % args.batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()

            # log
            iterations += 1
            if iterations % args.log_interval == 0:
                print(f"#epoch:{epoch}, #batch:{i + 1}, #iteration:{iterations}, loss:{loss}")

        # save model
        if (epoch + 1) % args.checkpoint_interval == 0:
            torch.save(fmnet.state_dict(), os.path.join(args.save_dir, 'epoch{}.pth'.format(epoch)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch the training of FMNet model."
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("-bs", "--batch-size", type=int, default=8, help="size of the batches")
    parser.add_argument("--n-epochs", type=int, default=50, help="number of epochs of training")
    parser.add_argument("-nv", "--n-vertices", type=int, default=2500, help="Number of vertices used per shape")
    parser.add_argument("-nb", "--num-blocks", type=int, default=7, help="number of resnet blocks")
    parser.add_argument('-d', '--dataroot', required=False, default="../data/faust/processed",
                        help='root directory of the dataset')
    parser.add_argument('--save-dir', required=False, default="../data/save/", help='root directory of the dataset')
    parser.add_argument("--n-cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument('--no-cuda', action='store_true', help='Disable GPU computation')
    parser.add_argument("--checkpoint-interval", type=int, default=5, help="interval between model checkpoints")
    parser.add_argument("--log-interval", type=int, default=1, help="interval between logging train information")

    args = parser.parse_args()
    train_fmnet(args)
