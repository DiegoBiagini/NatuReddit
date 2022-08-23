import tensorflow_model_analysis as tfma
from typing import List

def custom_eval_shared_model( eval_saved_model_path, model_name, eval_config, **kwargs, ) -> tfma.EvalSharedModel: 
    print(eval_saved_model_path)
    print(model_name)
    print(eval_config)

    return None


def custom_extractors( eval_shared_model, eval_config, tensor_adapter_config, ) -> List[tfma.extractors.Extractor]:
    print(eval_shared_model)
    print(eval_config)
    print(tensor_adapter_config)
    return None