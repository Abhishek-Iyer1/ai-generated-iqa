import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import seaborn as sns

from scipy import stats


def word_function(string: str):
    words = len(string.split(" "))
    return words


if __name__ == "__main__":
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
    unique_prompt_df = data_df["prompt"].drop_duplicates().copy(deep=True)
    prompt_mos_df = data_df[["name","prompt", "mos_quality", "mos_align"]].copy(deep=True)

    # Calculate necessary statistics for plotting
    length_of_prompts_char = unique_prompt_df.apply(len)
    length_of_prompts_words = unique_prompt_df.apply(word_function)
    prompt_mos_df["prompt_len"] = prompt_mos_df["prompt"].apply(len)
    prompt_mos_df["model_name"] = prompt_mos_df["name"].apply(lambda x: x.split("_")[0] + " " + x.split("_")[1])
    
    # Calculate mean and std of both
    char_mean = np.mean(length_of_prompts_char)
    char_std = np.std(length_of_prompts_char)
    words_mean = np.mean(length_of_prompts_words)
    words_std = np.std(length_of_prompts_words)

    analysis_df = pd.DataFrame({
        "words_dist" : length_of_prompts_words,
        "char_dist" : length_of_prompts_char
    })
    fig = plt.figure(figsize=(10,6))
    # plt.tight_layout()
    nrows = len(prompt_mos_df["model_name"].unique()) // 2
    ncols = 2
    for i, model_name in enumerate(prompt_mos_df["model_name"].unique()):
        model_subset_df = prompt_mos_df[prompt_mos_df["model_name"] == model_name]
        srocc = stats.spearmanr(model_subset_df["mos_align"], model_subset_df["prompt_len"])
        fig.add_subplot(nrows, ncols, i+1)
        sns.scatterplot(data=model_subset_df, x="prompt_len", y="mos_quality", size="mos_align", hue="mos_align")
        plt.title(f"{model_name} - srocc: {srocc.statistic:.2f}, pvalue: {srocc.pvalue:.2f}")
    plt.show()