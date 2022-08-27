torch-model-archiver \
    --model-name naturedditTI \
    --version 1.0 \
    --model-file code/pipelines/model/full_model.py \
    --serialized-file saved_models/1661280802/model_weight.pt \
    --extra-files saved_models/1661280802/model_cfg.yaml,saved_models/1661280802/label_mapping.yaml \
    --export-path torchserve/models \
    -f \
    --handler code/pipelines/serve_handler.py