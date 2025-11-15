import numpy as np
import tqdm
import matplotlib.pyplot as plt
import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from normalized_NN_dataset import Wine_QT
from NN_MLP import MLP

import json
import os

def get_arguments():
    parser = argparse.ArgumentParser(description='MLP Practice')
    parser.add_argument('--data_path', type=str, required=False, default='WineQT.csv',help='')

    parser.add_argument('--bs', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--max_epoch', type=int, default=200, help='')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    parser.add_argument('--gpu', type=int, default=0, help='gpu')

    return parser.parse_args()


if __name__ == "__main__":

    args = get_arguments()

    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    train_data = Wine_QT(
        data_path=args.data_path,
        split="train"
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.bs,
        shuffle=True,
        drop_last=False
    )

    num_feat = train_data.get_num_feat
    feat_values = train_data.get_class_values
    num_classes = len(feat_values)
    # print(f"Class values: {feat_values.astype(int).tolist()}")
    # print(f"Number of Classes: {num_classes}")
    mlp = MLP(
        feat_in=num_feat,
        feat_out = num_classes,
        dropout=args.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    model_optim = torch.optim.Adam(mlp.parameters(), lr=args.lr)

    # folder for saved models
    os.makedirs("models", exist_ok=True)
    # train
    train_loss_list = []
    min_loss = np.inf
    for epoch in tqdm.tqdm(range(args.max_epoch)):
        mlp.train() # set the model to train mode

        epoch_loss = 0

        for i, (feat, label) in enumerate(train_loader):

            model_optim.zero_grad()

            feat = feat.float().to(device)
            label = label.long().to(device)

            pred = mlp(feat)

            loss = criterion(pred, label)

            epoch_loss += loss.item()

            loss.backward()
            model_optim.step()

        train_loss_list.append(epoch_loss)

        model_name = f"NN_mlp_bs{args.bs}_lr{args.lr}_drop{args.dropout}_ep{args.max_epoch}.pth"

        if epoch_loss < min_loss:
            min_loss = epoch_loss
            torch.save(
                mlp.state_dict(),
                os.path.join("models", model_name)
            )

            result = {
            "batch_size": args.bs,
            "learning_rate": args.lr,
            "dropout": args.dropout,
            "max_epoch": args.max_epoch,
            "loss": min_loss,
            "best_model": model_name
            }

            # append to a results.json
            if os.path.exists("results.json"):
                with open("results.json", "r") as f:
                    results_list = json.load(f)
            else:
                results_list = []

            results_list.append(result)

            with open("results.json", "w") as f:
                json.dump(results_list, f, indent=4)

        print(">>> Epoch: {}  Loss: {}".format(epoch+1, epoch_loss))

    plt.plot(range(len(train_loss_list)), train_loss_list)
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy")
    plt.savefig("training_loss.png")

    print(f"Lowest loss: {min_loss}")



