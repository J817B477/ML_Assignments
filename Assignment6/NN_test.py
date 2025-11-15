import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from normalized_NN_dataset import Wine_QT
from NN_MLP import MLP

import json
import os

import pandas as pd

# def get_arguments():
#     parser = argparse.ArgumentParser(description='MLP Practice')
#     parser.add_argument('--data_path', type=str, required=False, default='WineQT.csv',help='')

#     parser.add_argument('--bs', type=int, default=1, help='batch size')
#     parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

#     parser.add_argument('--gpu', type=int, default=0, help='gpu')

#     return parser.parse_args()

if __name__ == "__main__":

    # args = get_arguments()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    test_data = Wine_QT(
        data_path="WineQT.csv",
        split="test"
    )

    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        drop_last=True
    )

    # loads json for appending
    with open("results.json", "r") as f:
            results = json.load(f)

    lookup = {d["best_model"]: d for d in results}

    model_dir = "models/"
    for model_file in os.listdir(model_dir):

        entry = lookup[model_file]

        # init + load model
        dropout = float(entry["dropout"])
        mlp = MLP(
            feat_in=test_data.get_num_feat,
            feat_out=len(test_data.get_class_values),
            dropout=dropout
        ).to(device)

        mlp.load_state_dict(torch.load(os.path.join(model_dir, model_file), map_location=device))
        mlp.eval()

        # testing loop
        gt = []
        preds = []
        probs = []
        total_loss = 0
        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for feat, label in test_loader:
                feat = feat.float().to(device)
                label = label.long().to(device)
                out = mlp(feat)
                loss = criterion(out, label)
                total_loss += loss.item()
                prob = F.softmax(out, dim=1)
                _, p = torch.max(prob, 1)
                gt.append(int(label.item()))
                probs.append(prob.squeeze(0).cpu().numpy().tolist())
                preds.append(int(p.item()))

        acc = accuracy_score(gt, preds)
        precision = precision_score(gt, preds, average="macro")
        recall = recall_score(gt, preds, average="macro")
        cm = confusion_matrix(gt, preds).tolist()

        # update existing dict
        entry.update({
            "test_loss": total_loss,
            "test_accuracy": acc,
            "test_precision": precision,
            "test_recall": recall,
            "predictions": preds,
            "confusion_matrix": cm,
            "probabilities": probs,
            "true_labels": gt,
            "model_path": os.path.join(model_dir, model_file)
        })

    with open('results.json', "w") as f:
        json.dump(results, f, indent=4)
    

    ##### parsing results ####

    results_df = pd.DataFrame(results)
    results_df.to_csv("test_train_results.csv", index=False)

        


