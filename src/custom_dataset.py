import torch
import pandas as pd
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset
from modules.CLIP import clip

class AGIQA(Dataset):

    def __init__(self, df: pd.DataFrame, device):
        self.image_paths = df["full_paths"]
        self.labels = df["mos_quality"]
        self.df_copy = df
        self.model, self.compose = clip.load("RN50", device=device)
        self.prompts = df["prompt"]

    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, float]:
        
        # Load Image from path based on index
        # image = Image.open(self.image_paths.iloc[index])
        image = self.compose(Image.open(self.image_paths.iloc[index])).unsqueeze(0)
        image_features = self.model.encode_image(image)
        
        # Load Prompt
        prompt = clip.tokenize(self.prompts.iloc[index])
        prompt_features = self.model.encode_text(prompt)

        # Load MOS Quality score
        mos_quality = self.labels.iloc[index]


        # # Compose a PIL to tensor transform
        # pil_to_tensor = transforms.Compose([
        #     transforms.PILToTensor()
        # ])

        # # Convert to a tensor
        # image_tensor: torch.Tensor = pil_to_tensor(image)

        # Return Tensor and respective MOS_quality score
        return (image.float(), prompt_features, float(mos_quality))

    def get_information(self, index) -> pd.Series:
        # Return the correct row from the dataframe which contains all necessary information
        return self.df_copy.loc[index]