"""Main executer script.
Reads a config file to train the CNN models
"""
import configparser
import os

import pl_bolts
import pytorch_lightning as pl
import torch  # needed for `optim=eval(config['trainer']['optim']))`
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb
################################
# Import models and datamodules
################################
from datamodules.datafolders import DataFolders
from models.models import ResNets
################################
# custom callback
from utils.image_prediction_logger import ImagePredictionLogger

################################


def main(config_file):
    """Main function.
    """
    # read in config file
    config = configparser.ConfigParser()
    config.read(config_file)
    ################################
    pl.seed_everything(10)

    print(f"pl version: {pl.__version__}")
    print(f"pl_bolts version: {pl_bolts.__version__}")

    wandb.login()

    ################################
    val_prop = float(config['trainer']['val_prop'])
    batch_size = int(config['trainer']['batch_size'])
    desired_batch_size = int(config['trainer']['desired_batch_size'])

    dset_mean = eval(config['dataset']['dset_mean'])
    dset_std = eval(config['dataset']['dset_std'])

    dm = DataFolders(data_dir=config['dataset']['dset_dir'],
                     batch_size=batch_size,
                     val_prop=val_prop,
                     dset_mean=dset_mean,
                     dset_std=dset_std,
                     five_crop=eval(config['model']['five_crop'])
                     )
    dm.prepare_data()
    dm.setup()

    print("Dataloader setup complete")

    # grab samples to log predictions on
    samples = next(iter(dm.val_dataloader()))

    ################################
    wandb.init(
        project=config['logger']['project'],
        group=config['logger']['group_name'],
        config={s: dict(config.items(s)) for s in config.sections()},
    )
    wandb.run.name = config['logger']['run_name']
    wandb.run.save()

    # create new early stopping
    early_stopping = EarlyStopping(
        monitor='valid/loss_epoch',
        min_delta=1e-6,
        patience=int(config['trainer']['patience']),
        verbose=True,
        mode='min'
    )

    # create checkpoint
    checkpoint_callback = ModelCheckpoint(dirpath='./',
                                          filename='best_weights',
                                          save_top_k=1,
                                          verbose=False,
                                          monitor='valid/loss_epoch',
                                          mode='min'
                                          )

    # make sure no "best_weights" files are stored:
    for fname in os.listdir():
        if fname.startswith("best_weights"):
            print(f'Warning: removing {fname}')
            os.remove(fname)

    wandb_logger = WandbLogger(sync_step=False)

    trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=int(config['trainer']['num_gpus']),
        max_epochs=int(config['trainer']['max_epochs']),
        deterministic=True,      # keep it deterministic
        gradient_clip_val=2.5,   # clip large gradients
        stochastic_weight_avg=True,
        accumulate_grad_batches=int(desired_batch_size /\
                                    batch_size),
        callbacks=[ImagePredictionLogger(samples, dm.idx_to_class),
                   early_stopping,
                   checkpoint_callback]
    )

    print(f"ResNet model, pretrained {eval(config['model']['pretrained'])}")
    # setup model
    if os.path.isfile(config['model']['load_weights_path']):
        load_weights_path = os.path.normpath(
            config['model']['load_weights_path'])
    else:
        load_weights_path = False

    model = ResNets(in_dims=(3, 224, 224),
                    lr=float(config['trainer']['lr']),
                    n_classes=len(dm.classes),
                    model_filename=config['logger']['model_name'],
                    class_names=[k for k, _ in dm.class_to_idx.items()],
                    resnet_layers=int(config['model']['resnet_layers']),
                    pretrained=eval(config['model']['pretrained']),
                    load_weights_path=load_weights_path,
                    freeze=eval(config['trainer']['freeze']),
                    unfreeze=int(config['trainer']['unfreeze']),
                    optim=eval(config['trainer']['optim']),
                    five_crop=eval(config['model']['five_crop']))

    # fit the model
    trainer.fit(model, dm)

    # evaluate the model on a test set
    trainer.test(datamodule=dm,
                 ckpt_path='best_weights.ckpt')  # uses best-saved model

    wandb.finish()

    # save checkpoint as torch file
    os.replace('best_weights.ckpt', config['logger']['model_name'])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-cf', '--config_file')

    args = parser.parse_args()
    if not args.config_file:
        main('config.ini')
    elif os.path.isfile(args.config_file):
        main(os.path.normpath(args.config_file))
    else:
        print('Please check provided config file')
