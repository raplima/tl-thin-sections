"""
Definition of classes for the Pytorch Lightning models.
"""

import numpy as np
import pl_bolts
import pytorch_lightning as pl
import torch
import wandb
from pl_bolts.models.self_supervised import SwAV
from torch import nn
from torch.nn import functional as F
from torchvision import models


class LitBaseModel(pl.LightningModule):
    """Overriding pl.LightningModule for experiments.
    """

    def __init__(self, in_dims,
                 n_classes=10,
                 class_names=None,
                 lr=1e-4,
                 model_filename=None,
                 optim=None
                 ):
        """

        Args:
            in_dims (list): input dimensions.
            n_classes (int, optional): Number of output neurons/classes. Defaults to 10.
            class_names (list, optional): Class names used. Defaults to None.
            lr (float, optional): Optimizer learning rate. Defaults to 1e-4.
            model_filename (string, optional): Full path for onnx model to be saved. Defaults to None.
        """

        super().__init__()

        # the base model has n_layers:
        n_layer_1 = 128
        n_layer_2 = 256

        # we flatten the input Tensors and pass them through an MLP
        self.layer_1 = nn.Linear(np.prod(in_dims), n_layer_1)
        self.layer_2 = nn.Linear(n_layer_1, n_layer_2)
        self.layer_3 = nn.Linear(n_layer_2, n_classes)

        # log hyperparameters
        self.save_hyperparameters()

        # compute the accuracy using PyTorch Lightning
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def forward(self, x):
        """Forward step for the model.

        Args:
            x (torch tensor): input tensor.

        Returns:
            logits (tensor): computed logits by the model.
        """
        batch_size, *dims = x.size()

        # stem: flatten
        x = x.view(batch_size, -1)

        # learner: two fully-connected layers
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))

        # task: compute class logits
        logits = self.layer_3(x)

        return logits

    def loss(self, xs, ys):
        """Compute the loss for the batch.

        Args:
            xs (torch tensor): Input tensor.
            ys (torch tensor): Label tensor.

        Returns:
            [logits, loss]: computed logits and loss
        """
        logits = self(xs)  # this calls self.forward

        loss = F.cross_entropy(logits, ys)
        return logits, loss

    def training_step(self, batch, batch_idx):
        """pl training step.

        Args:
            batch (torch tensor): batch
            batch_idx (int): batch index

        Returns:
            torch: computed loss
        """
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)

        # logging metrics we calculated by hand
        self.log('train/loss', loss, on_epoch=True)
        # logging a pl.Metric
        self.train_acc(preds, ys)
        self.log('train/acc', self.train_acc, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """pl validation step.

        Args:
            batch (torch tensor): batch
            batch_idx (int): batch index

        Returns:
            logits: computed logits
        """
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)

        self.valid_acc(preds, ys)

        # default on val/test is on_epoch only
        self.log("valid/loss_epoch", loss)
        self.log('valid/acc_epoch', self.valid_acc)

        return logits

    def validation_epoch_end(self, validation_step_outputs):
        """pl validation_epoch_end

        Args:
            validation_step_outputs (list of tensors): outputs from validation_step
        """

        flattened_logits = torch.flatten(torch.cat(validation_step_outputs))
        self.logger.experiment.log(
            {"valid/logits": wandb.Histogram(flattened_logits.to("cpu")),
             "global_step": self.global_step})

    def test_step(self, batch, batch_idx):
        """pl test step.

        Args:
            batch (torch tensor): batch
            batch_idx (int): batch index

        Returns:
            list(ys, preds): true labels, predicted labels
        """
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)

        self.test_acc(preds, ys)
        self.log("test/loss_epoch", loss, on_step=False, on_epoch=True)
        self.log("test/acc_epoch", self.test_acc, on_step=False, on_epoch=True)

        return ys.cpu(), preds.cpu()

    def test_epoch_end(self, test_step_outputs):
        """pl test epoch end

        Args:
            test_step_outputs (list of tensors): outputs from test_step
        """
        dummy_input = torch.zeros(self.hparams["in_dims"], device=self.device)

        if self.hparams["model_filename"]:
            model_filename = f'{self.hparams["model_filename"]}.onnx'
        else:
            model_filename = "model_final.onnx"
        torch.onnx.export(self, dummy_input, model_filename)

        res = np.array(test_step_outputs)

        ys = torch.cat(res[:, 0].tolist()).cpu().detach().numpy()
        preds = torch.cat(res[:, 1].tolist()).cpu().detach().numpy()

        self.logger.experiment.log(
            {"test/confmatrix": wandb.plot.confusion_matrix(probs=None,
                                                            y_true=ys,
                                                            preds=preds,
                                                            class_names=self.hparams["class_names"]
                                                            ),
             })

    def configure_optimizers(self):
        if self.hparams['optim']:
            return self.hparams['optim'](self.parameters(), lr=self.hparams["lr"])
        else:
            return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])

######################################################################################
######################################################################################


class ResNets(LitBaseModel):
    """Setup ResNet models based on LitBaseModel
    """

    def __init__(self,
                 in_dims,
                 n_classes=10,
                 class_names=None,
                 lr=1e-3,
                 model_filename=None,
                 resnet_layers=18,
                 pretrained=True,
                 freeze=False,
                 unfreeze=5,
                 optim=None):
        """
        Args:
            in_dims (list): input dimensions.
            n_classes (int, optional): Number of output neurons/classes. Defaults to 10.
            class_names (list, optional): Class names used. Defaults to None.
            lr (float, optional): Optimizer learning rate. Defaults to 1e-4.
            model_filename (string, optional): Full path for onnx model to be saved. Defaults to None.
            resnet_layers (int, optional): Number of ResNet layers. Defaults to 18.
            pretrained (bool, optional): Trained on ImageNet. Defaults to True.
            freeze (bool, optional): Freeze the backbone for the initial epochs. Defaults to False. 
            unfreeze (int, optional): Epoch to unfreeze the backbone (when freeze=True). Defaults to 10.
        """

        super().__init__(in_dims=in_dims,
                         n_classes=n_classes,
                         class_names=class_names,
                         lr=lr,
                         model_filename=model_filename,
                         optim=optim
                         )

        self.save_hyperparameters()

        self.model = {
            18: models.resnet18(pretrained=pretrained),
            34: models.resnet34(pretrained=pretrained),
            50: models.resnet50(pretrained=pretrained)
        }[resnet_layers]

        # fc becomes a hidden layer
        self.model.fc = nn.Linear(self.model.fc.in_features, 1024)
        self.finetune_layer = nn.Linear(1024, n_classes)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

            # make sure hidden layer can be trained:
            for param in self.model.fc.parameters():
                param.requires_grad = True
            for param in self.finetune_layer.parameters():
                param.requires_grad = True

    def forward(self, x):
        """Forward step for the model.

        Args:
            x (torch tensor): input tensor.

        Returns:
            logits (tensor): computed logits by the model.
        """

        if len(x.size()) < 4:
            x = torch.unsqueeze(x, 0)

        # logits = self.model(x)

        # return logits

        x = self.model(x)
        x = F.relu(x)
        logits = self.finetune_layer(x)

        return logits

    def on_epoch_end(self):
        # a hook is cleaner (but a callback is much better)
        if self.hparams["freeze"]:
            self.model.train()
            if self.trainer.current_epoch == self.hparams["unfreeze"]:
                for param in self.model.parameters():
                    param.requires_grad = True
