import shutil
import datetime
from typing import Dict, List, Tuple
import requests
import logging
import time
from pathlib import Path
import pandas as pd
import praw
from PIL import Image
import io

def get_raw_submissions(subreddits : List[str], date_start : datetime, date_end : datetime, pushshift_url : str):
    # Build get request
    get_data = {}
    get_data["subreddit"] = ','.join(subreddits)
    get_data["before"] = int(time.mktime(date_end.timetuple()))
    get_data["after"] = int(time.mktime(date_start.timetuple()))
    get_data["size"] = 100

    aggregated = None
    # Send it
    try:
        # Get the timestamp for the last submission obtained
        # Keep sending requests with increasing 'after'
        while True:
            response = requests.get(pushshift_url, params=get_data)
            response.raise_for_status()

            out = response.json()

            if not out["data"]:
                return aggregated
            else:
                if aggregated is None:
                    aggregated = out
                else:
                    aggregated["data"] += out["data"]

            last_timestamp = out["data"][-1]["created_utc"]
            get_data["after"] = last_timestamp + 1 

    except requests.HTTPError as e:
        logging.error(f"HTTP error: {e}")
    return None

def get_filtered_submissions(subreddits : List[str], date_start : datetime, date_end : datetime, clean_params : Dict, pushshift_url : str):
    raw_submissions = get_raw_submissions(subreddits, date_start, date_end, pushshift_url)
    if raw_submissions is None:
        raise EnvironmentError(f"No submissions were obtained, Request: subreddit={subreddits}, date_start={date_start}, date_end={date_end}")
    raw_sub_list = raw_submissions["data"]
    filter_keys = lambda d : {k:d[k] for k in clean_params}

    return list(map(filter_keys, raw_sub_list))

def scan_reddit_create_ds(subreddits : List[str], clean_params : List[str], 
                            timespan : Tuple[datetime.datetime, datetime.datetime], data_folder : Path, img_extensions : List[str],
                            pushshift_url : str, praw_connection : praw.Reddit) -> Tuple[Path, Path]:
    """
    Returns a tuple containing the paths to: (output_csv, image_folder)
    """
    date_start, date_end = timespan

    submissions = get_filtered_submissions(subreddits=subreddits, date_start=date_start, date_end=date_end, 
                                            clean_params=clean_params, pushshift_url=pushshift_url)

    # Give a name to the record and create dir
    record_name = date_start.strftime("%Y-%m-%d") + "_" + date_end.strftime("%Y-%m-%d")
    folder_path = data_folder / record_name
    folder_path.mkdir(parents=True, exist_ok=True)

    # Create dataframe
    df = pd.DataFrame(columns = clean_params)
    # Download images and add record to df
    logging.info(f"Obtained {len(submissions)} submissions")
    for subm in submissions:
        img_url = subm["url"]

        # Take only the first image from a gallery (I don't trust galleries)
        if "reddit.com/gallery/" in img_url:
            submission = praw_connection.submission(url = img_url)
            try:
                image_dict = submission.media_metadata
            except AttributeError:
                continue
            if image_dict is None:
                continue
            # Take the first image [0], the biggest resolution [s], the url [u]
            img_url = list(image_dict.values())[0]['s']['u']
            img_name = img_url[img_url.rfind("/") + 1:img_url.rfind("?")]
        else:
            url_extension = img_url[img_url.rfind(".")+1:]
            if url_extension not in img_extensions:
                logging.warning(f"Not considered : {img_url}")
                continue 
            img_name = img_url[img_url.rfind("/") + 1:]

        try:
            response = requests.get(img_url, stream=True)
            # If image size is less than 5kb skip it, it's either broken or too small
            content = response.content
            if len(content) < 5 * 1024:
                continue

            image = Image.open(io.BytesIO(content))
            image.save(folder_path / img_name)

            del response
            subm["url"] = img_url
            subm["img_name"] = img_name
            df = pd.concat([df, pd.DataFrame([subm])], ignore_index=True)
        except:
            logging.error(f"Could not download:{img_url}")
    
    df["created_utc"] = df["created_utc"].map(lambda x : datetime.datetime.fromtimestamp(x).isoformat())
    logging.info(f"Of these {len(df)} were valid")

    csv_path = str(folder_path) + ".csv"
    df.to_csv(csv_path)
    return (Path(csv_path), folder_path)


