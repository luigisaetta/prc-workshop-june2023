#!/usr/bin/env python
# coding: utf-8

# ### Image Classification with MNIST dataset and Lightning
# 
# In this second NB the FC NN has been replaced with a **CNN**.
# 
# There is an improvement: accuracy goes to **0.99** !!
# 
# * added checkpoint to save best based on val_loss

# In[1]:


import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import torch

from IPython.core.display import display

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms

# from here we get MNIST dataset
from torchvision.datasets import MNIST

# In[2]:


# where we're storing the downloaded datase
PATH_DATASETS = "."

BATCH_SIZE = 256 if torch.cuda.is_available() else 64


# In[3]:


class LitMNISTCNN(LightningModule):
    def __init__(self, data_dir=PATH_DATASETS, learning_rate=2e-4):

        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.learning_rate = learning_rate

        # dataset specific attributes
        self.num_classes = 10
        # shape of input images in MNIST
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # normalization is clarified here
                # https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # Define PyTorch model: a simple CNN
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Flatten(),
            nn.Linear(3*3*64, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.num_classes),
        )

        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def forward(self, x):
        # the model outputs logits not probabilities
        # this is better for numerical stability
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    # dataset and dataloaders

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=BATCH_SIZE, num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=BATCH_SIZE, num_workers=6)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=BATCH_SIZE, num_workers=6)


# In[4]:


checkpoint_callback = ModelCheckpoint(dirpath="checkpoint_mnist", save_top_k=2, monitor="val_loss",
                                     mode="min",
                                     filename="mnist-{epoch:02d}-{val_loss:.2f}")


# In[5]:


model = LitMNISTCNN()

trainer = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,
    max_epochs=20,
    callbacks=[TQDMProgressBar(refresh_rate=20), checkpoint_callback],
    logger=CSVLogger(save_dir="logs/"),
)


# #### Fit

# In[6]:


trainer.fit(model)


# In[7]:


trainer.test()




