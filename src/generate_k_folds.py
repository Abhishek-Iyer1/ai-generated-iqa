import pandas as pd
import os
import pickle

from custom_dataset import AGIQA
from sklearn.model_selection import KFold, train_test_split

def generate_k_fold_splits():

    # Set paths for dataset and image prefixs
    data_path = "../data/AGIQA-3k/AGIQA-3k/data.csv"
    image_prefix = "../data/AGIQA-3k/AGIQA-3k/Images/"

    # Load dataframe, prepend image prefixs and generate dataset
    data_df = pd.read_csv(data_path)
    data_df["full_paths"] = data_df["name"].apply(lambda x: os.path.join(image_prefix, x))
    agiqa = AGIQA(data_df)

    # Make K fold splits, store train, val, test indices in a dict by fold.
    k_fold = 5
    val_split = 0.1
    equiv_val_split = val_split / ((k_fold - 1)/k_fold)

    kf = KFold(n_splits=k_fold, shuffle=True)
    index_dict = {}

    for i, (train_index, test_index) in enumerate(kf.split(X=agiqa.image_paths, y=agiqa.labels)):
        
        train_index_new, val_index = train_test_split(train_index, test_size=equiv_val_split)

        index_dict[f"Fold {i+1}"] = {}
        index_dict[f"Fold {i+1}"]["Train Index"] = train_index_new
        index_dict[f"Fold {i+1}"]["Val Index"] = val_index
        index_dict[f"Fold {i+1}"]["Test Index"] = test_index

        # print(len(index_dict[f"Fold {i+1}"]["Train Index"]), len(index_dict[f"Fold {i+1}"]["Val Index"]), len(index_dict[f"Fold {i+1}"]["Test Index"]))

    return index_dict

def save_dict(index_dict : dict):

    with open('../data/k_fold_splits.pkl', 'wb') as f:
        pickle.dump(index_dict, f)
    
    return

if __name__ == "__main__":
    index_dict = generate_k_fold_splits()
    save_dict(index_dict)

