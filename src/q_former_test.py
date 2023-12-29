import os
import torch
import pickle
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import sys

sys.path.append("/home/smart/Abhishek-Iyer1/dev_ws/ai-generated-iqa/")

from tqdm import tqdm
from custom_dataset import AGIQA
from generate_k_folds import generate_k_fold_splits
from model import load_model
from torch.utils.data import DataLoader
from scipy import stats
from torch.utils.tensorboard import SummaryWriter

from modules.CLIP import clip
from PIL import Image
from transformers import Blip2QFormerConfig, Blip2QFormerModel

def test_qformer():

    # Create SummaryWriter instance that will log to ./runs/ by default
    writer = SummaryWriter()

    # Set device to be used
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set paths for dataset and image prefixs
    data_path = "../data/AGIQA-3k/AGIQA-3k/data.csv"
    image_prefix = "../data/AGIQA-3k/AGIQA-3k/Images/"
    split_path = "../data/CDS_0.csv"

    # Load dataframe, prepend image prefixs and generate dataset
    data_df = pd.read_csv(data_path)
    data_df["full_paths"] = data_df["name"].apply(lambda x: os.path.join(image_prefix, x))
    discard_models_condition = ((data_df["name"].str.startswith("AttnGAN")) | (data_df["name"].str.startswith("glide")))

    filtered_df = data_df[(np.bitwise_not(discard_models_condition))]

    split_df = pd.read_csv(split_path)
    
    train_prompts = split_df.query("split_0 == 'train'")["prompt"] 
    val_prompts = split_df.query("split_0 == 'val'")["prompt"] 
    test_prompts =  split_df.query("split_0 == 'test'")["prompt"]

    train_df = filtered_df[filtered_df["prompt"].isin(train_prompts)]
    val_df = filtered_df[filtered_df["prompt"].isin(val_prompts)]
    test_df = filtered_df[filtered_df["prompt"].isin(test_prompts)]

    # Load Model
    config = Blip2QFormerConfig()
    qformer_model = Blip2QFormerModel(config)

    # Create Sub Datasets from indices
    # training_dataset = AGIQA(train_df)
    # val_dataset = AGIQA(val_df)
    test_dataset = AGIQA(test_df, device)

    # Create DataLoaders to take advantage of batching, multiprocessing, and shuffling
    fold = "Fold 1"
    batch_size = 32
    # training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Run a simple training loop where weights are set to 0, both losses updated, weights updated. (Per Epoch)
    epochs = 5
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(qformer_model.parameters(), lr=0.001, momentum=0.9)

    for  image_features, text_features, mos_scores in tqdm(test_dataloader):

        image_features = image_features.to(device)
        text_features = text_features.to(device)

        query_outputs = qformer_model(
            query_embeds=query_tokens,
            encoder_hidden_states=image_features,
            encoder_attention_mask=image_attention_mask,
            return_dict=return_dict,
        )
        print(image_features.shape, text_features.shape)
        

if __name__ == "__main__":
    test_qformer()
