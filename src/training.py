import os
import torch
import pickle
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from custom_dataset import AGIQA
from generate_k_folds import generate_k_fold_splits
from model import load_model
from torch.utils.data import DataLoader
from scipy import stats
from torch.utils.tensorboard import SummaryWriter

def run_training_pipeline():

    # Create SummaryWriter instance that will log to ./runs/ by default
    writer = SummaryWriter()

    # Set device to be used
    device = 'cuda'

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

    num_folds_to_run = 2
    subset_keys = list(index_dict)[0:num_folds_to_run]
    subset_dict = dict((k, index_dict[k]) for k in subset_keys)

    # Run a for loop for each fold where you use indices split to load train, val, and test for each fold
    for fold, (train_index, val_index, test_index) in tqdm(subset_dict.items()):
        
        print(f"Training {fold}...")

        # Load the model in order to reset its weights
        my_resnet = load_model().to(device)

        # Create Sub Datasets from indices
        training_dataset = AGIQA(data_df.iloc[subset_dict[fold][train_index]])
        val_dataset = AGIQA(data_df.iloc[subset_dict[fold][val_index]])
        test_dataset = AGIQA(data_df.iloc[subset_dict[fold][test_index]])

        # Create DataLoaders to take advantage of batching, multiprocessing, and shuffling
        batch_size = 32
        training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # Run a simple training loop where weights are set to 0, both losses updated, weights updated. (Per Epoch)
        epochs = 5
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(my_resnet.parameters(), lr=0.001, momentum=0.9)
        my_resnet.train()

        for epoch in tqdm(range(0, epochs)):
            
            train_epoch_loss = 0

            # Run training loop for one epoch
            for x_batch, y_batch in tqdm(training_dataloader):

                x_batch = x_batch.float().to(device)
                y_batch = y_batch.reshape([y_batch.size()[0],1]).float().to(device)

                optimizer.zero_grad()
                y_pred = my_resnet(x_batch)
                # print(f"Predicitons: {y_pred}, Ground Truth: {y_batch}")
                loss = loss_fn.forward(y_pred, y_batch)
                loss.backward()
                optimizer.step()

                train_epoch_loss += loss.item()
            
            writer.add_scalar(f"{fold}/Loss/train", train_epoch_loss, epoch+1)

            print(f"Train Loss Epoch {epoch+1}: {train_epoch_loss}")

            #Initialize valid loss
            valid_epoch_loss = 0
            # Set model to eval mode
            my_resnet.eval()
            val_srocc = 0

            for val_x, val_y in tqdm(val_dataloader):
                val_x = val_x.float().to(device)
                val_y = val_y.float().to(device)

                y_pred = my_resnet.forward(val_x)

                val_batch_loss = loss_fn.forward(y_pred, val_y)

                valid_epoch_loss += val_batch_loss

                val_srocc += stats.spearmanr(y_pred.detach().cpu(), val_y.detach().cpu()).statistic

            writer.add_scalar(f"{fold}/Loss/valid", valid_epoch_loss, epoch+1)
            writer.add_scalar(f"{fold}/Acc/val_srocc", ((val_srocc * batch_size) / len(val_dataset)), epoch+1)

            print(f"Valid Loss Epoch {epoch+1}: {valid_epoch_loss}, Val SROCC Accuracy Average: {(val_srocc * batch_size) / len(val_dataset)}")

            test_epoch_loss = 0
            srocc = 0
            for x_test, y_test in tqdm(test_dataloader):
                x_test: torch.Tensor = x_test.to(device)
                y_test: torch.Tensor = y_test.reshape([y_test.size()[0],1]).to(device)
                y_pred = my_resnet.forward(x_test)
                loss = loss_fn.forward(y_pred, y_test)
                test_epoch_loss += loss
                srocc += stats.spearmanr(y_pred.detach().cpu(), y_test.detach().cpu()).statistic
            
            writer.add_scalar(f"{fold}/Loss/test", test_epoch_loss, epoch+1)
            writer.add_scalar(f"{fold}/Acc/srocc", ((srocc * batch_size) / len(test_dataset)), epoch+1)

            print(f"Test Loss: {test_epoch_loss}, SROCC Accuracy Average: {(srocc * batch_size) / len(test_dataset)}")

        # Save Model and Weights
        torch.save(my_resnet.state_dict(), f'model/my_resnet_{fold}.pth')

    # Flush the tensorboard information to save it to disk
    writer.flush()

    # Close writer
    writer.close()
    
if __name__ == "__main__":
    run_training_pipeline()