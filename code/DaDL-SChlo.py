import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset,DataLoader,random_split,ConcatDataset
# from prediction import Model
import os


class Model(nn.Module):
    def __init__(self, num_classes=5, init_weights=False):
        super(Model, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 10)),
            nn.ReLU(inplace=True),
        )
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=200, nhead=2)
        self.classifier = nn.Sequential(
            nn.Linear(200, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.squeeze(3)
        x = self.encoder_layer(x)
        x = torch.mean(x, axis=1)
        x = self.classifier(x)
        return x


import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset,DataLoader,random_split,ConcatDataset
# from prediction import Model
import os
import csv
from sklearn.metrics import accuracy_score

def Accuracy(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        p = sum(np.logical_and(y_true[i], y_pred[i]))
        q = sum(np.logical_or(y_true[i], y_pred[i]))
        count += p / q
    return count / y_true.shape[0]


def Precision(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        if sum(y_pred[i]) == 0:
            continue
        count += sum(np.logical_and(y_true[i], y_pred[i])) / sum(y_pred[i])
    return count / y_true.shape[0]


def Recall(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        if sum(y_true[i]) == 0:
            continue
        count += sum(np.logical_and(y_true[i], y_pred[i])) / sum(y_true[i])
    return count / y_true.shape[0]


def F1Measure(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        if (sum(y_true[i]) == 0) and (sum(y_pred[i]) == 0):
            continue
        p = sum(np.logical_and(y_true[i], y_pred[i]))
        q = sum(y_true[i]) + sum(y_pred[i])
        count += (2 * p) / q
    return count / y_true.shape[0]

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = pd.read_csv("feature_independent.csv").values


    data_list = data[1:, 5:]
    data_ndarry = np.array(data_list).astype("float64")
    data_lable = data[1:, :5].astype("float64")
    a = torch.tensor(data_ndarry)
    b = torch.tensor(data_lable)
    test_dataset = TensorDataset(a, b)
    val_loader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=True)
    pred_label = []
    true_label = []
    model = Model().to(device)
    weights_path = "../model/Model.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))
    val_bar = tqdm(val_loader)
    for val_data in val_bar:
        val_seq, val_labels = val_data
        # print(val_seq.shape)
        val_seq = val_seq.view(-1, 200, 10)
        val_seq = val_seq.float()
        outputs = model(val_seq.to(device))
        outputs = (outputs == outputs.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32)
        outputs = outputs.cpu().numpy()
        pred_label.extend(outputs)


    print(pred_label)

if __name__ == '__main__':
    main()
