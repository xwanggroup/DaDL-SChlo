import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
class CSVDataset(Dataset):
    def __init__(self, dir,label):
        self.dir = dir
        allData = pd.read_csv(self.dir)
        x_y = allData.values
        x = []

        for index, i in enumerate(x_y):
            if i[0] == float(label):
                print(np.array(i))
                x.append(np.array(i[1:]))
        self.dataset = np.array(x, dtype=np.float32)
            # np.random.shuffle(dataset)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def train_dataloader(path, batch_size=4, num_workers=0,label=1):
    dataloader = DataLoader(
        CSVDataset(path,label),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader