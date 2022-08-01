import json

from pandas import merge
import pendulum
from airflow.decorators import dag, task

from NatuReddit.code.data_service.pushshift_utils import scan_reddit_create_ds 
from NatuReddit.code.data_service.download_reddit_data import TimeUnits, get_timespan_upto
from NatuReddit.code.data_service.merge_datasets import merge_datasets
from NatuReddit.code.data_service.resize_images import resize_images

from pathlib import Path
import logging
import praw
import os
import subprocess

logger = logging.getLogger("airflow.task")

@dag(
    dag_id="daily_update",
    schedule_interval=None,
    start_date=pendulum.datetime(2022, 1, 1, tz="UTC"),
    catchup=False,
    tags=['data'],
)
def obtain_daily_data():
    """
    ### DAG which performs the operations necessary to expand the current unlabeled dataset. 
    Such operations should be performed each day.  
    Those are:  
    - Login to PRAW  
    - Obtain the posts from the last day  
    - Downsize the images as needed  
    - Merge them with the existing dataset  

    Then perform the operations necessary to store them in the cloud (dvc add and push, update .env)
    """
    # PATHS
    cwd =  Path("/opt/airflow/NatuReddit")

    main_folder = Path(__file__).parent.parent / 'NatuReddit'
    data_path =  main_folder / "data"
    settings_path = main_folder / "code/data_service/data_extraction_settings.json"

    # Obtain the currently latest dataset from the environment
    old_dataset_csv = data_path / os.getenv("LATEST_DS")

    @task()
    def dvc_pull_task(cwd : Path):
        os.chdir(cwd)
        stream = os.popen("""dvc pull -v
        ls data
        """)
        logger.info(stream.read())
        return 1

    @task(multiple_outputs=True)
    def read_login_settings_task(settings_path : str, prev=None):
        """
        #### Login with PRAW using the settings in the appropriate folder  
        Also returns the other settings read from the appropriate file
        """
        try:
            with open(settings_path, "rb") as f:
                settings = json.loads(f.read())

                try:
                    subreddits = settings["subreddits"]
                    to_extract = settings["to_extract"]
                    img_extensions = settings["img_extensions"]
                    pushshift_url = settings["pushshift_url"]

                    reddit_client_id = settings["reddit_client_id"]
                    reddit_client_secret = settings["reddit_client_secret"]
                except KeyError:
                    logger.error("Settings file is malformed")
                    raise KeyError
        except FileNotFoundError:
            logger.error(f"Settings file not found in {settings_path}")
            raise KeyError
    
        return settings
            

    @task(multiple_outputs=True)
    def obtain_posts_task(settings : dict, save_location : Path, prev=None):
        """
        #### Obtain posts from the last day
        """
        # Instantiate praw (needed for galleries)
        praw_connection = praw.Reddit(client_id=settings['reddit_client_id'],
                            client_secret=settings['reddit_client_secret'], 
                            user_agent="android:com.example.myredditapp:v1.2.3 (by u/kemitche)")

        last_day = get_timespan_upto(1, TimeUnits.DAY)
        out_csv_path, out_folder_path = scan_reddit_create_ds(settings['subreddits'], settings['to_extract'], last_day, save_location, 
                                settings['img_extensions'], settings['pushshift_url'], praw_connection)

        return {'out_csv':str(out_csv_path), 'out_folder':str(out_folder_path)}

    @task()
    def resize_images_task(folder_path : str, prev=None):
        """
        #### Resize downloaded images s.t. their long side is at most 1980 px
        """
        max_size = 1920
        resize_images(Path(folder_path), max_size)
        return 1

    @task()
    def merge_datasets_task(ds1 : str, ds2 : str, destination : Path, prev=None):
        """
        #### Merge the new daily update with the old dataset
        """

        merged_csv, merged_folder = merge_datasets(Path(ds1), Path(ds2), destination, remove_old=False)
        return {"merged_csv":str(merged_csv), "merged_folder":str(merged_folder)}

    @task()
    def dvc_update_task(new_csv_path : str, new_folder_path : str, prev=None):
        logger.info(new_csv_path)
        logger.info(new_folder_path)
        new_csv_name = new_csv_path[new_csv_path.rfind("/")+1:]
        new_folder_name = new_folder_path[new_folder_path.rfind("/")+1:]
        logger.info(new_csv_name)
        logger.info(new_folder_name)

        result = subprocess.run(f"dvc add {new_csv_path}", stdout=subprocess.PIPE)
        logger.info(result.stdout)

        stream = os.popen(f"""
        echo on
        dvc add {new_csv_path}
        dvc add {new_folder_path}
        sed -i 's/LATEST_DS=".*\.csv"/LATEST_DS="{new_csv_name}"/ airflow/Dockerfile
        git commit -a -m "Daily update"
        """)

        print(stream.read())
        os.environ["LATEST_DS"] = new_csv_name

        #logger.info(stream.read())
        return 1
    
    @task()
    def push_task(prev=None):
        
        # stream = os.popen(f"""
        # dvc push
        #git push
        # """)
        stream = os.popen("ls")
        logger.info(stream.read())
        logger.info("Pushed everything to dvc and github")
        return 1

    # Main DAG flow
    dvc_pull_out = dvc_pull_task(cwd=cwd)

    settings = read_login_settings_task(settings_path=settings_path, prev=dvc_pull_out)

    posts_out = obtain_posts_task(settings, data_path, prev=None)
    
    out_folder_path = posts_out['out_folder']
    out_csv_path = posts_out['out_csv']

    resize_out = resize_images_task(out_folder_path)

    merge_out = merge_datasets_task(old_dataset_csv, out_csv_path, data_path, prev=resize_out)

    dvc_update_out = dvc_update_task(new_csv_path=merge_out["merge_csv"], new_folder_path=merge_out["merge_folder"])

    push_out = push_task(dvc_update_out)



obtain_data_dag = obtain_daily_data()