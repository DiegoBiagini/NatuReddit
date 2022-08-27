from typing import Dict
from transformers import ResNetModel, BertModel, AutoFeatureExtractor, AutoTokenizer
from torchmetrics import Accuracy
import yaml
from torch import nn
from torch.utils.data import DataLoader
import torch
import logging
import numpy as np
import mlflow
from torch.nn.utils.rnn import pad_sequence
import json

class FullModel(nn.Module):

    def __init__(self, cfg_file = None, cfg = None, inference=False, train_device = None) -> None:
        super().__init__()
        if not cfg is None:
            self.cfg = cfg
        elif not cfg_file is None:
            with open(cfg_file, "r") as f:
                cfg = yaml.safe_load(f)
                self.cfg = cfg

        else:
            #  Default config
            cfg = {}
            cfg["model"] = {}
            cfg["train"] = {}

            cfg["model"]["bert_model"] = "prajjwal1/bert-tiny"
            cfg["model"]["resnet_model"] = "microsoft/resnet-18"
            cfg["model"]["out_classes"] = 3
            cfg["model"]["projection_dim"] = 512

            cfg["train"]["n_epochs"] = 20
            cfg["train"]["batch_size"] = 16

            cfg["train"]["opt"] = {}
            cfg["train"]["opt"]["name"] = "adam"
            cfg["train"]["opt"]["lr"] = 0.001
            cfg["train"]["opt"]["betas"] = [0.9, 0.999]
            cfg["train"]["opt"]["eps"] = 1e-8
            cfg["train"]["opt"]["weight_decay"] = 0

            self.cfg = cfg

        self.cfg = cfg
        model_cfg = cfg["model"]
        
        # Log every param into mlflow
        for p in model_cfg:
            mlflow.log_param(p, model_cfg[p])
        for p in cfg["train"]:
            if type(cfg["train"][p]) == dict:
                for pr in cfg["train"][p]:
                    mlflow.log_param(p+"_"+pr, cfg["train"][p][pr])
            else:
                mlflow.log_param(p, cfg["train"][p])

        # Works for resnet models
        self.cnn_encoder = ResNetModel.from_pretrained(model_cfg["resnet_model"])
        self.cnn_encoder_out_size = self.cnn_encoder.config.hidden_sizes[-1]

        self.cnn_feature_extractor = AutoFeatureExtractor.from_pretrained(model_cfg["resnet_model"])

        # Works for bert models
        self.bert_encoder = BertModel.from_pretrained(model_cfg["bert_model"])
        self.bert_encoder_out_size = self.bert_encoder.config.hidden_size

        self.bert_tokenizer = AutoTokenizer.from_pretrained(model_cfg["bert_model"])

        self.classifier_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=(self.cnn_encoder_out_size+self.bert_encoder_out_size), out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=model_cfg["projection_dim"], out_features=model_cfg["out_classes"])
        )
        self.inference = inference
        self.train_device = train_device
        if not self.train_device is None:
            self.to(train_device)


    def forward(self, img, text):
        x_img = torch.flatten(self.cnn_encoder(img).pooler_output, start_dim=1)

        x_text = self.bert_encoder(text).pooler_output

        x = torch.cat([x_img, x_text], dim=1)
        x = self.classifier_head(x)

        if self.inference:
            x = torch.softmax(x,-1)
        return x

    def train_model(self, train_ds, val_ds):
        logging.info(f"Start training model with settings:\n {json.dumps(self.cfg, indent=4)}")
        # Instantiate dataloaders
        batch_size = self.cfg["train"]["batch_size"]

        training_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        validation_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        # Loss
        loss_fn = nn.CrossEntropyLoss()
        accuracy_fn = Accuracy(num_classes=3, average="micro")

        # Optimizer
        if self.cfg["train"]["opt"]["name"] == "adam":
            opt_cfg = self.cfg["train"]["opt"]
            # I don't think this works
            #optimizer = torch.optim.Adam(**opt_cfg)
            optimizer = torch.optim.Adam(
                params=self.parameters(),
                lr = float(opt_cfg["lr"]),
                betas=opt_cfg["betas"],
                eps = float(opt_cfg["eps"]),
                weight_decay=float(opt_cfg["weight_decay"])
            )

        else:
            # Default optimizer
            optimizer = torch.optim.SGD(lr=0.1)

        for epoch in range(1, self.cfg["train"]["n_epochs"]+1):
            if self.train_device == 'cuda':
                torch.cuda.empty_cache()
            ## Train
            train_epoch_loss, train_epoch_acc = self.train_one_epoch(training_loader, optimizer, loss_fn, accuracy_fn)
            if self.train_device == 'cuda':
                torch.cuda.empty_cache()
            ## Validation
            out_metrics = self.eval_model(validation_loader, req_metrics={"loss" : loss_fn, "accuracy": accuracy_fn}, device = self.train_device)
            print("test")
            logging.info(f"""Epoch:{epoch}/{self.cfg['train']['n_epochs']} \n
                Train set: Accuracy:{train_epoch_acc}, Loss{train_epoch_loss}\n 
                Validation set: Accuracy:{out_metrics["accuracy"]}, Loss{out_metrics["loss"]}""")

            # Log into mlflow
            mlflow.log_metric("train_loss", train_epoch_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_epoch_acc, step=epoch)

            mlflow.log_metric("validation_loss",out_metrics["loss"], step=epoch)
            mlflow.log_metric("validation_accuracy", out_metrics["accuracy"], step=epoch)
    
    def train_one_epoch(self, training_loader, optimizer, loss_fn, accuracy_fn):
        self.train()

        train_epoch_loss = []
        train_epoch_acc = []
        for i, data in enumerate(training_loader):
            text_input, img_input, labels = data

            if not self.train_device is None:
                img_input = img_input.to(self.train_device)
                text_input = text_input.to(self.train_device)
                labels = labels.to(self.train_device)

            optimizer.zero_grad()

            # Forward
            outputs = self.forward(img_input, text_input)

            # Backward
            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            y_pred = torch.softmax(outputs, dim=-1).cpu()

            train_epoch_loss += [loss.item()]
            train_epoch_acc += [accuracy_fn(y_pred, labels.cpu()).item()]

        train_epoch_loss = np.mean(train_epoch_loss)
        train_epoch_acc = np.mean(train_epoch_acc)
        return train_epoch_loss, train_epoch_acc
    
    def eval_model(self, eval_dl, req_metrics : Dict[str, callable] = None, device = None, log_mlflow = False):
        metrics = {m : [] for m in req_metrics}
        self.eval()

        for i, data in enumerate(eval_dl):
            text_input, img_input, labels = data

            if not device is None:
                img_input = img_input.to(self.train_device)
                text_input = text_input.to(self.train_device)
                labels = labels.to(self.train_device)

            outputs = self.forward(img_input, text_input)
            out_logits = torch.softmax(outputs, dim=-1)
            # Compute metrics
            for m in req_metrics:
                # Use logits for most metrics
                if m == 'loss':
                    corr_output = outputs
                else:
                    corr_output = out_logits

                out_m = req_metrics[m](corr_output.cpu(), labels.cpu())
                if type(out_m) == torch.Tensor:
                    out_m = out_m.item()
                metrics[m] += [out_m]

        for m in metrics:
            metrics[m] = np.mean(metrics[m])     
        if log_mlflow:   
            for m in metrics:
                mlflow.log_metric("eval_"+m, metrics[m])
        return metrics

def collate_fn(data):
    titles, imgs, labels = zip(*data)
    padded_titles = pad_sequence(titles, batch_first=True)
    return padded_titles, torch.stack(imgs), torch.stack(labels)