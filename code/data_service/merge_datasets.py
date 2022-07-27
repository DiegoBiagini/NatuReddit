from genericpath import exists
import logging
from argparse import ArgumentError, ArgumentParser
from pathlib import Path
import pandas as pd
import datetime
import shutil
from functools import partial

def merge_datasets(ds1 : Path, ds2 : Path, dst : Path, remove_old : bool):
    df1 = pd.read_csv(ds1, index_col=0)
    df2 = pd.read_csv(ds2, index_col=0)

    # Create a new df which contains all records between the two dates
    def get_start_end_datetime(full_str : str):
        start_date_str = full_str[:full_str.find("_")]
        end_date_str = full_str[full_str.find("_")+1:]
        start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d")
        return (start_date, end_date)

    start_date1, end_date1 = get_start_end_datetime(str(ds1.stem))
    start_date2, end_date2 = get_start_end_datetime(str(ds2.stem))

    new_start_date = start_date1 if start_date1 < start_date2 else start_date2
    new_end_date = end_date1 if end_date1 > end_date2 else end_date2

    new_df = pd.concat([df1, df2], ignore_index=True).drop_duplicates(["id", "img_name"])

    # Create new folder and copy elements to it
    format = "%Y-%m-%d"
    new_name = new_start_date.strftime(format)+ "_" + new_end_date.strftime(format)

    new_folder = dst / new_name
    if new_folder.exists():
        shutil.rmtree(new_folder)
    new_folder.mkdir()

    def copy_to_folder(row : pd.Series, origin_folder : Path, dst_folder : Path):
        filename = row["img_name"]
        new_file =  dst_folder / filename
        # If two images have the same name I guess they are the same
        if not new_file.exists():
            shutil.copy(origin_folder / filename, new_file)

    df1.apply(partial(copy_to_folder, origin_folder=ds1.parent / ds1.stem, dst_folder=new_folder), axis=1)
    df2.apply(partial(copy_to_folder, origin_folder=ds2.parent / ds2.stem, dst_folder=new_folder), axis=1)

    new_df.to_csv(dst / (new_name + ".csv"))

    # Remove the original datasets
    if remove_old:
        origin_folder1 = ds1.parent / ds1.stem
        origin_folder2 = ds2.parent / ds2.stem
        
        shutil.rmtree(origin_folder1)
        shutil.rmtree(origin_folder2)

        ds1.unlink()
        ds2.unlink()



def validate_dataset(ds : Path) -> bool:
    str_ds = str(ds)
    if not ds.exists():
        logging.error(f"{ds} does not exist")
        return False
    if not str_ds.endswith('.csv'):
        logging.error(f"{ds} is not a csv file")
        return False
    try:
        str_ds = str(ds.stem)
        start_date_str = str_ds[:str_ds.find("_")]
        end_date_str = str_ds[str_ds.find("_")+1:]
        start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d")
        if end_date < start_date:
            raise Exception
    except Exception as e:
        logging.error(f"{ds} does not have the correct time format (yyyy-mm-dd_yyyy-mm-dd)")
        return False
    correct_folder = ds.parent.absolute() / ds.stem
    if not correct_folder.exists():
        logging.error(f"Image folder missing for {ds}")
        return False
    return True

if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)

    # Read command line arguments

    parser = ArgumentParser(description='Merge two datasets, a datasets is defined as a .csv file and a folder of images in the same folder')

    parser.add_argument('dataset1', metavar='D1', type=Path, help='Path to first csv file')
    parser.add_argument('dataset2', metavar="D2", type=Path, help="Path to second csv file")
    parser.add_argument('destination', metavar="D", type=Path, help="Path where to save the dataset", nargs="?", default=Path.cwd())
    parser.add_argument('-r', action='store_true', help="If toggled on removes the two original datasets after merging")

    args = vars(parser.parse_args())

    if not validate_dataset(args['dataset1']):
        logging.error(f"Dataset1 in {args['dataset1']} was malformed")
        exit()
    if not validate_dataset(args['dataset2']):
        logging.error(f"Dataset2 in {args['dataset2']} was malformed")       
        exit() 

    merge_datasets(args['dataset1'], args['dataset2'], args['destination'], args['r'])
