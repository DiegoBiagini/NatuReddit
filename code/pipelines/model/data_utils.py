from torch.utils.data import Dataset
import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
import torchvision
from PIL import Image
import torch
from torch.nn.utils.rnn import pad_sequence

class NatuRedditDS(Dataset):

    def __init__(self, og_ds, img_location, text_preprocess_fn = None, img_preprocess_fn = None) -> None:
        super().__init__()
        
        # Get the columns
        for el in og_ds.take(1):
            cols = el.keys()
        new_df = pd.DataFrame(columns=cols) 

        for el in og_ds:
            el_numpy = {k: np.squeeze(el[k].numpy()) for k in el}
            
            inc_df = pd.DataFrame.from_dict(el_numpy)

            if len(new_df) == 0:
                new_df = inc_df
            else:
                new_df = pd.concat([new_df, inc_df])
        self.new_df = new_df.drop(list(cols)[0], axis=1)
        self.img_location = img_location
        self.text_preprocess_fn = text_preprocess_fn
        self.img_preprocess_fn = img_preprocess_fn

        # Define the mapping between class labels and categorical encoding
        labels = new_df["choice"].unique()
        self.label_to_cat_dict = {j.decode('utf-8'):i for i,j in enumerate(labels)}
        self.cat_to_label_dict = {self.label_to_cat_dict[k]:k for k in self.label_to_cat_dict.keys()}
    
    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, idx):
        # Get the row in the ds
        row = self.new_df.iloc[idx]
        # Get the title and preprocess it
        raw_title = row["title"].decode('utf-8')
        if not self.text_preprocess_fn is None:
            title = torch.as_tensor(self.text_preprocess_fn(raw_title)['input_ids'])
        else:
            title = raw_title

        # Get the image and preprocess it
        full_url = Path(self.img_location) / row["img_name"].decode('utf-8')
        pil_img = Image.open(full_url)

        if not self.img_preprocess_fn is None:
            prepr_img = self.img_preprocess_fn(pil_img)['pixel_values'][0]
            img = torch.as_tensor(prepr_img)
        else:
            img = torchvision.transforms.ToTensor()(pil_img)
        # Get the label
        label = row["choice"].decode('utf-8')
        return title, img, torch.as_tensor(self.label_to_cat_dict [label])


def collate_fn(data):
    titles, imgs, labels = zip(*data)
    padded_titles = pad_sequence(titles, batch_first=True)
    return padded_titles, torch.stack(imgs), torch.stack(labels)