import os
from typing import Optional, Text, List
from absl import logging
from ml_metadata.proto import metadata_store_pb2
import tfx.v1 as tfx
from tfx.components.example_gen.csv_example_gen.component import CsvExampleGen
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.components.evaluator.component import Evaluator
from tfx.components.schema_gen.component import SchemaGen
from tfx.components.trainer.component import Trainer
from tfx.dsl.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor
import tfx.proto.example_gen_pb2 as epb2
from tfx.proto import trainer_pb2
from tfx.proto.pusher_pb2 import PushDestination
from tfx.components.pusher.component import Pusher
from pathlib import Path
import yaml
import tensorflow_model_analysis as tfma

PIPELINE_NAME = 'full_pipeline'
PIPELINE_ROOT = os.path.join('.', 'full_pipeline_output')
METADATA_PATH = os.path.join('.', 'tfx_metadata', PIPELINE_NAME, 'metadata.db')
DATA_PATH = str(Path(__file__).parent /  'data')
SERVING_MODEL_DIR = "saved_models/"
ENABLE_CACHE = True

def create_pipeline(
  pipeline_name: Text,
  pipeline_root:Text,
  enable_cache: bool,
  metadata_connection_config: Optional[
    metadata_store_pb2.ConnectionConfig] = None,
  beam_pipeline_args: Optional[List[Text]] = None
):
    print(os.getenv('MLFLOW_TRACKING_URI'))
    # Load pipeline config
    with open(Path(__file__).parent / "main_pipeline_cfg.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    components = []

    ############
    # ExampleGen

    output = epb2.Output(
        split_config=epb2.SplitConfig(splits=[
        epb2.SplitConfig.Split(name='train', hash_buckets=8),
        epb2.SplitConfig.Split(name='eval', hash_buckets=1),
        epb2.SplitConfig.Split(name='test', hash_buckets=1)
    ]))
    example_gen = CsvExampleGen(input_base=DATA_PATH, output_config=output)
    components.append(example_gen)

    ############
    # StatisticsGen
    stats_gen = StatisticsGen(example_gen.outputs["examples"])
    components.append(stats_gen)

    ############
    # SchemaGen
    schema_gen = SchemaGen(
        statistics=stats_gen.outputs['statistics'],
        infer_feature_shape=True)
    components.append(schema_gen)

    ###########
    # Trainer
    
    # Load model config here 
    with open(Path(__file__).parent / "model/full_model_cfg.yaml") as f:
        model_cfg = yaml.safe_load(f)
    trainer = Trainer(
        module_file=str(Path(__file__).parent /  "model/run_full_model.py"),
        custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(),
        eval_args=trainer_pb2.EvalArgs(),
        custom_config={"dataset_location" : cfg["dataset_location"], "model_cfg":model_cfg}
        )

    components.append(trainer)

    ###########
    # Evaluator
    """
    # Just don't, there is 0 documentation
    # You could export to ONNX and then ONNX -> tf savedmodel
    # But then you can't preprocess, it would be a mess with strings

    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key='choice')],
        slicing_specs=[tfma.SlicingSpec()])

    model_analyzer = Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      eval_config=eval_config,
      module_file=str(Path(__file__).parent /  "model/run_evaluator.py"))

    components.append(model_analyzer)
    """
    ########
    # Pusher
    pusher = Pusher(
      model=trainer.outputs['model'],
      push_destination=PushDestination(
          filesystem=PushDestination.Filesystem(
              base_directory=SERVING_MODEL_DIR)))

    components.append(pusher)

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=enable_cache,
        metadata_connection_config=metadata_connection_config,
        beam_pipeline_args=beam_pipeline_args, 
    )

def run_pipeline():
    ppl = create_pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        enable_cache=ENABLE_CACHE,
        metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(METADATA_PATH)
    )

    tfx.orchestration.LocalDagRunner().run(ppl)

if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  run_pipeline()
