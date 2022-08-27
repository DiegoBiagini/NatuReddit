from ts.torch_handler.base_handler import BaseHandler
import torch
import os
from full_model import FullModel
import yaml
from PIL import Image
import io

CFG_FILE_NAME = "model_cfg.yaml"
MAPPING_FILE_NAME = "label_mapping.yaml"

class NatuRedditHandler(BaseHandler):

    def __init__(self):
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        self.initialized = True
        #  load the model, refer 'custom handler class' above for details
                #  load the model
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        print("Model path", model_pt_path)

        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")
        
        # Load extra files
        # Apparently extra files are not registered into the manifest, nice
        cfg_file = os.path.join(model_dir, CFG_FILE_NAME)
        mapping_file = os.path.join(model_dir, MAPPING_FILE_NAME)

        with open(mapping_file, "r") as f:
            self.label_to_cat = yaml.safe_load(f)
        self.cat_to_label = {self.label_to_cat[k]: k for k in self.label_to_cat}

        self.model = FullModel(cfg_file=cfg_file, inference=True)
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_pt_path))
        else:
            self.model.load_state_dict(torch.load(model_pt_path, map_location=torch.device("cpu")))

        self.initialized = True

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready
        # Text data comes in in the "text" field
        prepr_text = torch.as_tensor(self.model.bert_tokenizer(data[0]["text"].decode("utf-8"))['input_ids'])
        # Image comes in in the "image field"
        pil_img = Image.open(io.BytesIO(data[0]["image"]))
        prepr_img = torch.as_tensor(self.model.cnn_feature_extractor(pil_img)['pixel_values'][0])

        return {"text":prepr_text, "image":prepr_img}


    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        print("Image", model_input["image"].shape)
        #print("Text", model_input["text"].shape)
        model_output = self.model.forward(torch.unsqueeze(model_input["image"],0), torch.unsqueeze(model_input["text"],0))
        print("Model output", model_output)

        return model_output 

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        pred = torch.argmax(inference_output, dim=-1).numpy()
        print("Pred", pred)
        postprocess_output = self.cat_to_label[pred[0]]
        print("Final output", postprocess_output)
        return postprocess_output

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return [self.postprocess(model_output)]