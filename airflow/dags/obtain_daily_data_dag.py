import json
import pendulum
from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator
from ...code.data_service.pushshift_utils import scan_reddit_create_ds 
from ...code.data_service.download_reddit_data import TimeUnits, get_timespan_upto
from ...code.data_service.resize_images import resize_images
from ...code.data_service.merge_datasets import merge_datasets
from pathlib import Path
import logging
import praw
import os

@dag(
    dag_id="daily_update",
    schedule_interval=None,
    start_date=pendulum.datetime(2022, 1, 1, tz="UTC"),
    catchup=False,
    tags=['data'],
)
def obtain_daily_data():
    """
    ### DAG which performs the operations necessary to expand the current unlabeled dataset
    Such operations should be performed each day
    Those are:
    -Login to PRAW
    -Obtain the posts from the last day
    -Downsize the images as needed
    -Merge them with the existing dataset

    Then perform the operations necessary to store them in the cloud (dvc add and push, update .env)
    """

    @task(multiple_outpus=True)
    def praw_login_task(settings_path : str):
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
                    logging.error("Settings file is malformed")
                    raise KeyError
        except FileNotFoundError:
            logging.error("Settings file not found")
            raise KeyError
    
        # Instantiate praw (needed for galleries)
        praw_connection = praw.Reddit(client_id=reddit_client_id,
                            client_secret=reddit_client_secret, 
                            user_agent="android:com.example.myredditapp:v1.2.3 (by u/kemitche)")
        conn = {'praw_connection' : praw_connection}
        return {**conn, **settings}
            

    @task(multiple_outputs=True)
    def obtain_posts_task(settings : dict, save_location : Path):
        """
        #### Obtain posts from the last day
        """
        last_day = get_timespan_upto(1, TimeUnits.DAY)
        out_paths = scan_reddit_create_ds(settings['subreddits'], settings['to_extract'], last_day, save_location, 
                                settings['img_extensions'], settings['pushshift_url'], settings['praw_connection'])

        return out_paths

    @task()
    def resize_images_task(folder_path : Path):
        """
        #### Resize downloaded images s.t. their long side is at most 1980 px
        """
        max_size = 1920
        resize_images(folder_path, max_size)

    @task()
    def merge_datasets_task(ds1 : Path, ds2 : Path, destination : Path):
        """
        #### Merge the new daily update with the old dataset
        """

        merge_datasets(ds1, ds2, destination, remove_old=True)




    data_path =  main_folder / "data"
    main_folder = Path(__file__).parent.parent
    settings_path = "code/data_service/data_extraction_settings.json"

    # Obtain the currently latest dataset from the environment
    old_dataset_csv = data_path / os.getenv("LATEST_DS")


    # Main DAG flow
    settings = praw_login_task(settings_path=main_folder / settings_path)

    out_csv_path, out_folder_path = obtain_posts_task(settings, data_path)
    resize_images_task(out_folder_path)

    merge_datasets_task(old_dataset_csv, out_csv_path, data_path)

    dvc_update_task = BashOperator(
        task_id="dvc_update",
        bash_command="""dvc add $NEW_DS
        dvc add $NEW_DS_FOLDER
        sed -i 's/LATEST_DS=.*\.csv/LATEST_DS=$NEW_DS/
        git commit -a -m "Daily update"
        """,
        env = {'NEW_DS':str(out_csv_path.name), "NEW_DS_FOLDER":str(out_folder_path.name)},
        append_env = True
    )

    resize_images >> merge_datasets >> dvc_update_task

    push_task = BashOperator(
        task_id="push",
        bash_command="""dvc push
        git push
        """
    )

    dvc_update_task >> push_task


obtain_data_dag = obtain_daily_data()