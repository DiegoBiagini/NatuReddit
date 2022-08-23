# A script just to move files from the unlabeled dataset into the labeled dataset
from genericpath import exists
import pandas as pd
from argparse import ArgumentParser
import logging
from pathlib import Path
import shutil

def move_ds(labeled_csv : Path, unlabeled_csv : Path, unlabeled_folder : Path):
    labeled_df = pd.read_csv(labeled_csv, index_col=0)
    
    labeled_folder = labeled_csv.parent / labeled_csv.stem
    if labeled_folder.exists():
        shutil.rmtree(labeled_folder)
    labeled_folder.mkdir(exist_ok=True)

    def copy_to_folder(filename):
        shutil.copyfile(unlabeled_folder / filename, labeled_folder / filename)

    labeled_df["img_name"].apply(copy_to_folder)



if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)

    # Read command line arguments

    parser = ArgumentParser(description='Move the images from an unlabeled dataset into a folder for the labeled ds')

    parser.add_argument('labeled', metavar='L', type=Path, help='Labeled csv')
    parser.add_argument('unlabeled', metavar="U", type=Path, help="Unlabeled csv")

    args = vars(parser.parse_args())

    unlabeled_csv : Path = args['unlabeled']
    unlabeled_folder = unlabeled_csv.parent / unlabeled_csv.stem
    if not unlabeled_folder.exists() or not unlabeled_csv.exists():
        logging.error(f"Unlabeled dataset {str(unlabeled_csv)} does not exist")
        exit()
    labeled_csv : Path = args['labeled']
    if not labeled_csv.exists():
        logging.error(f"Labeled dataset {str(labeled_csv)} does not exist")
        exit()
    
    move_ds(labeled_csv, unlabeled_csv, unlabeled_folder)