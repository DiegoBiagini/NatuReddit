import logging
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image

def resize_images(dst_folder : Path, max_size : int):
    img_extensions = ["png", "jpg", "jpeg"]

    filtered_files = []
    for ext in img_extensions:
        filtered_files.extend(dst_folder.glob("*."+ext))
    logging.info(f"{len(filtered_files)} images found")

    n_resize = 0
    for file in filtered_files:
        img = Image.open(file)
        w, h = img.size
        if w > h and w > max_size:
            new_w = max_size
            new_h = int(max_size*h/w)
        elif h > w and h > max_size:
            new_h = max_size
            new_w = int(max_size*w/h)
        else:
            continue
        img.resize((new_w, new_h)).save(file)
        n_resize += 1
    logging.info(f"Resized {n_resize} images")

if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)

    # Read command line arguments

    parser = ArgumentParser(description='Resize the images in a folder in such a way that their longest side is at most of a certain length')

    parser.add_argument('folder', metavar='F', type=Path, help='Folder containing the images')
    parser.add_argument('size', metavar="S", type=int, help="Max length of the long side", nargs=1)

    args = vars(parser.parse_args())

    dst_folder : Path = args['folder']
    if not dst_folder.exists():
        logging.error(f"Folder {str(dst_folder)} does not exist")
        exit()

    resize_images(dst_folder, args['size'][0])
