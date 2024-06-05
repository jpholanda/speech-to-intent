from trainer_config import NUM_WORKERS, get_arguments
from model import HubertSSLModel
from dataset import S2IDataset, collate_fn

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# SEED
SEED=100
pl.utilities.seed.seed_everything(SEED)
torch.manual_seed(SEED)

import os
os.environ['WANDB_MODE'] = 'online'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class LightningModel(pl.LightningModule):
    def __init__(self,):
        super().__init__()
        self.model = HubertSSLModel()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return [optimizer]

    def loss_fn(self, prediction, targets):
        return nn.CrossEntropyLoss()(prediction, targets)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)

        logits = self(x)        
        probs = F.softmax(logits, dim=1)
        loss = self.loss_fn(logits, y)

        winners = logits.argmax(dim=1)
        corrects = (winners == y)
        acc = corrects.sum().float()/float(logits.size(0))

        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return {
            'loss':loss, 
            'acc':acc
            }

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)

        logits = self(x)
        loss = self.loss_fn(logits, y)

        winners = logits.argmax(dim=1)
        corrects = (winners == y)
        acc = corrects.sum().float() / float( logits.size(0))

        self.log('val/loss' , loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc',acc, on_step=False, on_epoch=True, prog_bar=True)

        return {'val_loss':loss, 
                'val_acc':acc,
                }
        
    

if __name__ == "__main__":
    args = get_arguments()

    dataset = S2IDataset()

    train_len = int(len(dataset) * 0.90)
    val_len =  len(dataset) - train_len
    print(train_len, val_len)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(SEED))

    trainloader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=2, 
            shuffle=True, 
            num_workers=NUM_WORKERS,
            collate_fn = collate_fn,
        )
    
    valloader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=2, 
            num_workers=NUM_WORKERS,
            collate_fn = collate_fn,
        )

    model = LightningModel()

    run_name = "hubert-ssl"
    logger = WandbLogger(
        name=run_name,
        project='S2I-baseline'
    )

    model_checkpoint_callback = ModelCheckpoint(
            dirpath='checkpoints',
            monitor='val/acc', 
            mode='max',
            verbose=1,
            filename=run_name + "-epoch={epoch}.ckpt")

    trainer = Trainer(
            fast_dev_run=args.dev,
            gpus=1, 
            max_epochs=50, 
            checkpoint_callback=True,
            callbacks=[
                model_checkpoint_callback,
            ],
            logger=logger,
            )

    trainer.fit(model, train_dataloader=trainloader, val_dataloaders=valloader)

    