import torch
import torch.nn as nn
import pandas as pd
import os
import pickle 

from torch.utils.data import DataLoader
from scipy import stats
from model import load_model
from custom_dataset import AGIQA

def test():

    device = "cuda"

    # Set paths for dataset and image prefixs
    data_path = "../data/AGIQA-3k/AGIQA-3k/data.csv"
    image_prefix = "../data/AGIQA-3k/AGIQA-3k/Images/"

    # Load dataframe, prepend image prefixs and generate dataset
    data_df = pd.read_csv(data_path)
    data_df["full_paths"] = data_df["name"].apply(lambda x: os.path.join(image_prefix, x))
    agiqa = AGIQA(data_df)

    # Load split files from K folds or Run if file not present
    with open('../data/k_fold_splits.pkl', 'rb') as f:
        index_dict: dict = pickle.load(f)

    my_resnet = load_model()
    my_resnet.load_state_dict(torch.load('model/my_resnet_Fold 1.pth'))

    my_resnet.eval().to(device)
    loss_fn = nn.MSELoss()
    batch_size = 32
    for fold, (train_index, val_index, test_index) in index_dict.items():

        print(f"Testing {fold}...")
        test_dataset = AGIQA(data_df.iloc[index_dict[fold][test_index]])
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        test_loss = 0
        srocc = 0
        for x_test, y_test in test_dataloader:
            
            x_test: torch.Tensor = x_test.to(device)
            y_test: torch.Tensor = y_test.reshape([y_test.size()[0],1]).to(device)
            y_pred = my_resnet.forward(x_test)
            loss = loss_fn.forward(y_pred, y_test)
            test_loss += loss

            srocc += stats.spearmanr(y_pred.detach().cpu(), y_test.detach().cpu()).statistic
        print((srocc * batch_size) / len(test_dataset))
        print(srocc, test_loss, batch_size, len(test_dataset))

if __name__ == "__main__":
    test()


