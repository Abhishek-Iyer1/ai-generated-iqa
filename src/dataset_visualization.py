import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse

from PIL import Image

# Define paths
parser = argparse.ArgumentParser(description="Runs visualization pipeline for chosen dataset")
parser.add_argument('-d', '--dataset', choices=('agiqa-3k', 'aigciqa2023'), default='agiqa-3k', help="Choose a dataset to run the visualization pipeline on.")
args = parser.parse_args()

if args.dataset == "agiqa-3k":
    data_path = "../data/AGIQA-3k/AGIQA-3k/data.csv"
    image_prefix = "../data/AGIQA-3k/AGIQA-3k/Images/"
    img_path_col = "name"
    ncols = 5

elif args.dataset == "aigciqa2023":
    data_path = "../data/AIGCIQA2023/AIGCIQA2023/aigciqa_all_data.csv"
    image_prefix = "../data/AIGCIQA2023/AIGCIQA2023/allimg/"
    img_path_col = "im_loc"
    ncols = 6

else:
    # Code will never get here because of Argument Error raised by ArgParses
    pass

# Load DFs
data_df = pd.read_csv(data_path)
# print(data_df.head())
unique_prompt_df = data_df["prompt"].drop_duplicates().copy(deep=True)

# Pick prompt to display on random / through cli
random_index = random.randint(0, len(unique_prompt_df))

# Load all images belonging to that prompt and other relevant data
prompt_specifc_df: pd.DataFrame = data_df[data_df["prompt"] == unique_prompt_df.iloc[random_index]].copy(deep=True)
# print(prompt_specifc_df.head())

images = []
for r_index, row in prompt_specifc_df.iterrows():
    image_path = image_prefix + row[img_path_col]
    images.append(np.asarray(Image.open(image_path)))

prompt_specifc_df["image_arrays"] = images

# Plot all instances using matplotlib   
C = ncols
R = (len(prompt_specifc_df) // ncols) + np.clip(len(prompt_specifc_df) % ncols, 0, 1)
fig = plt.figure(figsize=(10,8))
plt.title(f"Prompt: {prompt_specifc_df['prompt'].iloc[0]}")
plt.tight_layout()
plt.axis("off")

for i in range(0, len(prompt_specifc_df)):
    # Image from Train set
    fig.add_subplot(R, C, i+1)
    plt.imshow(prompt_specifc_df["image_arrays"].iloc[i])
    plt.axis("off")

    if args.dataset == "agiqa-3k":
        name: str = prompt_specifc_df["name"].iloc[i]
        plt.title(f"Name: {name.split('_')[0] + '_' + name.split('_')[1]}\nmos_qual: {prompt_specifc_df['mos_quality'].iloc[i]:.2f}\nmos_align: {prompt_specifc_df['mos_align'].iloc[i]:.2f}")
    elif args.dataset == "aigciqa2023":
        name: str = prompt_specifc_df["model"].iloc[i]
        plt.title(f"Name: {name}\n") #Scene: {prompt_specifc_df['scene'].iloc[i]}\nmos_qual: {prompt_specifc_df['mosz1'].iloc[i]:.2f}\nsd1: {prompt_specifc_df['sd1'].iloc[i]:.2f

plt.show()