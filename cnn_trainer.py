import configparser
import os

import pl_bolts
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

################################
# Import models and datamodules
################################
from datamodules.datafolders import DataFolders
from models.models import ResNets
################################
# custom callback
from utils.image_prediction_logger import ImagePredictionLogger

################################


def main():
    # set up config file:
    # config = configparser.ConfigParser()
    # config['main'] = {'file_in': self.parent.lineEdit_inputFile.text(),
    #                   'file_out': self.parent.lineEdit_outputFile.text(),
    #                   'filter': 'vertical_derivative'}

    # config['filter'] = {'order': self.comboBox_order.currentText()}

    # # write config file:
    # with open('config.ini', 'w') as configfile:
    #     config.write(configfile)
    # config = configparser.ConfigParser()
    # config.read('config.ini')
    ################################
    dset_dir = '../data/Histological_images_MSI_vs_MSS'
    dset_mean = [0.7263, 0.5129, 0.6925]
    dset_std = [0.1444, 0.1833, 0.1310]
    resnet_layers = 50
    train_len = "500000"
    batch_size = 128
    max_epochs = 10
    # (torch.optim.RMSprop, 'RMSprop')
    optim = torch.optim.Adam
    lr = 1e-3
    pretrained = True
    pretrained_tag = 'randomly-initialized'
    num_gpus = -1
    run_tag = 'Adam1lr-3'
    load_weights_path = False  # 'trained_model.ckpt'  # False
    ################################
    pl.seed_everything(10)

    print(f"pl version: {pl.__version__}")
    print(f"pl_bolts version: {pl_bolts.__version__}")

    wandb.login()

    ################################
    # setup data
    train_len = int(train_len)
    val_len = int(round(0.2*train_len))
    dm = DataFolders(data_dir=dset_dir,
                     batch_size=batch_size,
                     train_len=train_len,
                     val_len=val_len,
                     dset_mean=dset_mean,
                     dset_std=dset_std)
    dm.prepare_data()
    dm.setup()

    print("Dataloader setup complete")

    # grab samples to log predictions on
    samples = next(iter(dm.val_dataloader()))

    ################################
    # optim = optim_with_tag[0]
    # optim_tag = optim_with_tag[1]

    config = dict(
        train_len=train_len,
        val_len=val_len,
        resnet_layers=resnet_layers,
        pretrained=pretrained,
        # pretrained can be any of the options below
        #pretrained = True,
        #pretrained = False,
        lr=lr,
        freeze=True,
        unfreeze=2,
        optim=optim,
        run_tag=run_tag,
        # optim and run number should be something like the ones below
        # optim = None,
        # run_number = '3-Adam 1e-3',
        # optim = torch.optim.RMSprop,
        # run_number = '3-RMSprop 1e-3',
    )

    group_tag = f"ResNet{config['resnet_layers']}_{pretrained_tag}"

    wandb.init(
        project="ts-tl",
        group=group_tag,
        config=config,
    )
    wandb.run.name = config['run_tag']
    wandb.run.save()
    wandb.log({'train_len': train_len,
               'valid_len': val_len})

    # create new early stopping
    early_stopping = EarlyStopping(
        monitor='valid/loss_epoch',
        min_delta=1e-6,
        patience=10,
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

    wandb_logger = WandbLogger(sync_step=False)

    trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=num_gpus,
        max_epochs=max_epochs,           # number of epochs
        deterministic=True,      # keep it deterministic
        callbacks=[ImagePredictionLogger(samples, dm.idx_to_class),
                   early_stopping,
                   checkpoint_callback]
    )

    print('ResNet model')
    # setup model
    model = ResNets(in_dims=(3, 224, 224),
                    lr=config['lr'],
                    n_classes=len(dm.classes),
                    model_filename=group_tag+run_tag,
                    class_names=[k for k, _ in dm.class_to_idx.items()],
                    resnet_layers=config['resnet_layers'],
                    pretrained=config['pretrained'],
                    freeze=config['freeze'],
                    unfreeze=config['unfreeze'],
                    optim=config['optim'])

    if load_weights_path:
        model = ResNets.load_from_checkpoint(load_weights_path)

    # fit the model
    trainer.fit(model, dm)

    # evaluate the model on a test set
    trainer.test(datamodule=dm,
                 ckpt_path='best_weights.ckpt')  # uses best-saved model

    wandb.finish()
    # save checkpoint as torch file
    os.rename('best_weights.ckpt', 'trained_model.ckpt')


if __name__ == "__main__":
    main()
