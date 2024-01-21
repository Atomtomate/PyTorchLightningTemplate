import os
import h5py
import numpy as np
from src.models.SimpleFC import SimpleFC_Lit
import lightning as L
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, StochasticWeightAveraging
import torch
torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":

    hparams = {"input_dim": 200,
            "fc_dims": [200, 200, 200, 200, 200],
            "dropout": 0.4,
            "output_dim": 200,
            "lr": 0.01,
            "batch_size": 128,
            "train_path": "data/julian/batch1.hdf5",
            # "val_path": "runs/example_run/data/data_val.h5",
            # "test_path": "runs/example_run/data/data_test.h5",
               }
    model = SimpleFC_Lit(hparams)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=50)
    val_ckeckpoint = ModelCheckpoint( # saved in `trainer.default_root_dir`/`logger.version`/`checkpoint_callback.dirpath`
            filename="{epoch}-{step}-{val_loss:.5f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            )
    callbacks = [val_ckeckpoint, lr_monitor, early_stopping]
    trainer = L.Trainer(max_epochs=100, accelerator="gpu", precision="16-mixed", callbacks=callbacks)
    trainer.fit(model)

    # # to load data
    # train_path = hparams["train_path"]
    # with h5py.File(train_path, "r") as hf:
    #     x = hf["Set1/GImp"][:]
    #     y = hf["Set1/SImp"][:]
    # print(x.shape)
    # print(y.shape)
    # print("x[0]", x[0])
    # # convert from complex to two real numbers and then concatenate
    # x = np.concatenate((x.real, x.imag), axis=1)
    # y = np.concatenate((y.real, y.imag), axis=1)
    #
    # print(x.shape)
    # print(y.shape)
    # print("x[0]", x[0])
    #
    #
