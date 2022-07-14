import shutil
import datetime
import string
from typing import Dict, List
import requests
import logging
import time
import json
from pathlib import Path
import pandas as pd

data_folder = Path(__file__).parent
img_extensions = ["png", "jpg", "jpeg"]

def get_raw_submissions(subreddits : List[str], date_start : datetime, date_end : datetime):
    pushshift_url = "https://api.pushshift.io/reddit/search/submission/"
    # Build get request
    get_data = {}
    get_data["subreddit"] = ','.join(subreddits)
    get_data["before"] = int(time.mktime(date_end.timetuple()))
    get_data["after"] = int(time.mktime(date_start.timetuple()))

    # Send it
    try:
        response = requests.get(pushshift_url, params=get_data)

        response.raise_for_status()
        return response.json()
    except requests.HTTPError as e:
        logging.error(f"HTTP error: {e}")
    return None

def get_filtered_submissions(subreddits : List[str], date_start : datetime, date_end : datetime, clean_params : Dict):
    raw_submissions = get_raw_submissions(subreddits, date_start, date_end)
    if raw_submissions is None:
        raise EnvironmentError(f"No submissions were obtained request: subreddit={subreddits}, date_start={date_start}, date_end={date_end}")
    raw_sub_list = raw_submissions["data"]
    filter_keys = lambda d : {k:d[k] for k in clean_params}

    return map(filter_keys, raw_sub_list)

def scan_reddit_create_ds(subreddits : List[str], clean_params : List[str]):
    date_start = datetime.datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    date_end = date_start + datetime.timedelta(days=1)

    submissions = get_filtered_submissions(subreddits=subreddits, date_start=date_start, date_end=date_end, clean_params=to_extract)

    # Give a name to the record and create dir
    record_name = date_start.strftime("%Y-%m-%d") + "_" + date_end.strftime("%Y-%m-%d")
    folder_path = data_folder / record_name
    folder_path.mkdir(parents=True, exist_ok=True)

    # Create dataframe
    df = pd.DataFrame(columns = clean_params)
    # Download images and add record to df
    for subm in submissions:
        img_url = subm["url"]
        
        url_extension = img_url[img_url.rfind(".")+1:]
        if url_extension not in img_extensions:
            logging.warning(f"Not considered : {img_url}")
            continue 
        try:
            response = requests.get(img_url, stream=True)
            img_name = img_url[img_url.rfind("/") + 1:]
            with open(folder_path / img_name, 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
            del response

            df = pd.concat([df, pd.DataFrame([subm])], ignore_index=True)
        except:
            logging.error(f"Could not download:{img_url}")
    df.to_csv(str(folder_path) + ".csv")


with open(Path(__file__).parent / "data_extraction_settings.json", "rb") as f:
    settings = json.loads(f.read())
    subreddits = settings["subreddits"]
    to_extract = settings["to_extract"]

scan_reddit_create_ds(subreddits, to_extract)