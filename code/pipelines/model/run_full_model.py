from tfx.components.trainer.fn_args_utils import FnArgs
import mlflow
from torch.utils.data import Dataset
from tensorflow_metadata.proto.v0 import schema_pb2
from tfx_bsl.tfxio import dataset_options
from tfx.utils.io_utils import parse_pbtxt_file
import tensorflow as tf
import pandas as pd
from data_utils import NatuRedditDS
from full_model import FullModel
from pathlib import Path
import torch
import logging
import shutil
import yaml

def run_fn(fn_args : FnArgs):
    logging.getLogger().setLevel(logging.INFO)
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    mlflow.start_run()
    schema = parse_pbtxt_file(fn_args.schema_file, schema_pb2.Schema())
    # Datasets which read from the tfrecord files
    train_ds_tf = fn_args.data_accessor.tf_dataset_factory(
        fn_args.train_files,
        dataset_options.TensorFlowDatasetOptions(batch_size=128, num_epochs=1),
        schema
    )

    val_ds_tf = fn_args.data_accessor.tf_dataset_factory(
        fn_args.eval_files,
        dataset_options.TensorFlowDatasetOptions(batch_size=128, num_epochs=1),
        schema
    )

    model = FullModel(cfg=fn_args.custom_config["model_cfg"], train_device=device)
    model = model.to(device)

    train_ds = NatuRedditDS(train_ds_tf, img_location = fn_args.custom_config["dataset_location"], 
        img_preprocess_fn=model.cnn_feature_extractor,
        text_preprocess_fn=model.bert_tokenizer)
    val_ds = NatuRedditDS(val_ds_tf, img_location = fn_args.custom_config["dataset_location"], 
        img_preprocess_fn=model.cnn_feature_extractor,
        text_preprocess_fn=model.bert_tokenizer)


    model.train_model(train_ds, val_ds)

    mlflow.end_run()

    target_folder = Path(fn_args.serving_model_dir)
    if not target_folder.exists():
        target_folder.mkdir()
    # Serialize model, model settings and label mapping
    with open(target_folder / "model_cfg.yaml", "w") as f:
        yaml.dump(fn_args.custom_config["model_cfg"], f)
    with open(target_folder / "label_mapping.yaml", "w") as f:
        yaml.dump(train_ds.label_to_cat_dict, f)
    torch.save(model.state_dict(), target_folder / "model_weight.pt")
    