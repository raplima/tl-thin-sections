import torch
import pytorch_lightning as pl
import wandb

class ImagePredictionLogger(pl.Callback):
    def __init__(self, val_samples, idx_to_class, num_samples=32):
        super().__init__()
        self.val_imgs, self.val_labels = val_samples
        self.val_imgs = self.val_imgs[:num_samples]
        self.val_labels = self.val_labels[:num_samples]

        self.idx_to_class = idx_to_class
          
    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)

        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)

        trainer.logger.experiment.log({
            "examples": [wandb.Image(x, 
                                     caption=f"Pred: {self.idx_to_class[pred.item()]}\nLabel: {self.idx_to_class[y.item()]}") 
                            for x, pred, y in zip(val_imgs, preds, self.val_labels)],
            "global_step": trainer.global_step
            })