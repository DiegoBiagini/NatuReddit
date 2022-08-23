from PIL import Image
import logging
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd

def remove_transparency(dataset_path, folder_path):
    df = pd.read_csv(dataset_path)

    filtered_files = []
    filtered_files.extend(folder_path.glob("*.png"))
    logging.info(f"Found {len(filtered_files)} PNG files")
    for file in filtered_files:
        png = Image.open(file).convert('RGBA')
        background = Image.new('RGBA', png.size, (255,255,255))

        alpha_composite = Image.alpha_composite(background, png).convert("RGB")
        new_file = Path(folder_path) / (str(Path(file).stem) + '.jpg')
        alpha_composite.save(new_file, 'JPEG', quality=90)

        Path(file).unlink()
        # Fix dataset 
        df["image_name"] = df["image_name"].replace(str(Path(file).name), new_file.name)
        logging.info(f"Replaced {file} with {new_file}")
    df.to_csv(dataset_path)


if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)

    # Read command line arguments

    parser = ArgumentParser(description='Remove the A channel in png images in a dataset (transforming them into jpgs)')

    parser.add_argument('dataset', metavar='D', type=Path, help='Dataset csv files containing the images')
    args = vars(parser.parse_args())

    dst_dataset : Path = args['dataset']
    dst_folder : Path = dst_dataset.parent / dst_dataset.stem
    if not dst_folder.exists() or dst_dataset.exists():
        logging.error(f"Da {str(dst_folder)} does not exist")
        exit()
    remove_transparency(dst_dataset, dst_folder)



