from enum import Enum, auto
import datetime
from dateutil.relativedelta import relativedelta
from argparse import ArgumentParser
from pathlib import Path
import json
import logging
from typing import Tuple
from .pushshift_utils import scan_reddit_create_ds
import praw

# Enum to set a timespan
class TimeUnits(Enum):
    YEAR = auto()
    MONTH = auto()
    WEEK = auto()
    DAY = auto()

def get_timespan_upto(n_of : int, unit : TimeUnits) -> Tuple[datetime.datetime, datetime.datetime]:
    now = datetime.datetime.today()
    if unit == TimeUnits.YEAR:
        delta = relativedelta(years=n_of)
    elif unit == TimeUnits.MONTH:
        delta = relativedelta(months=n_of)
    elif unit == TimeUnits.WEEK:
        delta = relativedelta(weeks=n_of)
    else:
        delta = relativedelta(days=n_of)
    start = now - delta
    return (start, now)

if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)

    # Read command line arguments

    parser = ArgumentParser(description='Ingest image and post data from a set of subreddits')

    parser.add_argument('n_of', metavar='N', nargs=1, type=int, help='The number of years/months/weeks/days up to which to get data')
    parser.add_argument('unit', metavar='U', nargs=1,
        choices=['years', 'months', 'weeks', "days"], help='Time unit to consider with N(years/months/weeks/days)')
    parser.add_argument("location", metavar="L", type=Path, help="Where to save the obtained data", nargs="?", default=Path(__file__).parent)

    args = vars(parser.parse_args())
    simplify_list = lambda x : x[0] if isinstance(x, list) else x
    args = {k:simplify_list(args[k]) for k in args}

    unit_dict = {"years": TimeUnits.YEAR, "months": TimeUnits.MONTH, "weeks": TimeUnits.WEEK, "days": TimeUnits.DAY}

    timespan = get_timespan_upto(args["n_of"], unit_dict[args["unit"]])

    # Read settings
    try:
        with open(Path(__file__).parent / "data_extraction_settings.json", "rb") as f:
            settings = json.loads(f.read())

            try:
                subreddits = settings["subreddits"]
                to_extract = settings["to_extract"]
                img_extensions = settings["img_extensions"]
                pushshift_url = settings["pushshift_url"]

                reddit_client_id = settings["reddit_client_id"]
                reddit_client_secret = settings["reddit_client_secret"]
            except KeyError:
                logging.error("Settings file is malformed")
                exit()
    except FileNotFoundError:
        logging.error("Settings file not found")
    
    # Instantiate praw (needed for galleries)
    praw_connection = praw.Reddit(client_id=reddit_client_id,
                         client_secret=reddit_client_secret, 
                         user_agent="android:com.example.myredditapp:v1.2.3 (by u/kemitche)")
                         
    scan_reddit_create_ds(subreddits, to_extract, timespan, args["location"], img_extensions, pushshift_url, praw_connection)