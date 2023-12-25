import torch
import pandas as pd
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset

class AGIQA(Dataset):

    def __init__(self, df: pd.DataFrame):
        self.image_paths = df["full_paths"]
        self.labels = df["mos_quality"]
        self.df_copy = df

    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, float]:
        
        # Load Image from path based on index
        image = Image.open(self.image_paths.iloc[index])
        
        # Load MOS Quality score
        mos_quality = self.labels.iloc[index]

        # Compose a PIL to tensor transform
        pil_to_tensor = transforms.Compose([
            transforms.PILToTensor()
        ])

        # Convert to a tensor
        image_tensor: torch.Tensor = pil_to_tensor(image)

        # Return Tensor and respective MOS_quality score
        return (image_tensor.float(), float(mos_quality))

    def get_information(self, index) -> pd.Series:
        # Return the correct row from the dataframe which contains all necessary information
        return self.df_copy.loc[index]