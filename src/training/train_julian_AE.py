from src.models.AE import AutoEncoder_01
from copy import deepcopy
import lightning as L
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, StochasticWeightAveraging, RichModelSummary, DeviceStatsMonitor
from lightning.pytorch.profilers import PyTorchProfiler
import torch
from pytorch_lightning.loggers import TensorBoardLogger
torch.set_float32_matmul_precision("highest")
torch.set_default_dtype(torch.float64)

hparams0 = {
            "model_name": "AE_nPrune_02_nLayers",
            "in_dim" : 200,
            "latent_dim": 30,
            "n_layers": 3,
            "dropout_in": 0.0,
            "dropout": 0.0,
            "with_batchnorm": False,
            "lr": 0.001584893192461114, #0.01,
            "batch_size": 1024,
            "train_path": "D:/data_batch2_nPrune.hdf5",
            "optimizer": "Adam",
            "activation": "ReLU",
            "SGD_weight_decay": 0.0,
            "SGD_momentum": 0.9,
            "SGD_dampening": 0.0,
            "SGD_nesterov": False,
            "loss": "MSE",
        }

hparams_list = [hparams0]
i = 0
if __name__ == "__main__":
        for hparams in hparams_list:
                for bs in [1024, 196, 32]:
                        hparams["batch_size"] = bs
                        for n_layers in [1,2,3,4,6,8,10,12,14,16,18,20,22,24,26,28,30]:
                                if i > 1:# or i == 6:
                                        hparams0["n_layers"] = n_layers
                                        model = AutoEncoder_01(hparams) 
                                        lr_monitor = LearningRateMonitor(logging_interval='step')
                                        early_stopping = EarlyStopping(
                                                monitor="val_loss",
                                                patience=40)
                                        logger = TensorBoardLogger("lightning_logs", name=hparams["model_name"])
                                        val_ckeckpoint = ModelCheckpoint( # saved in `trainer.default_root_dir`/`logger.version`/`checkpoint_callback.dirpath`
                                                filename="{epoch}-{step}-{val_loss:.8f}",
                                                monitor="val_loss",
                                                mode="min",
                                                save_top_k=2,
                                                save_last =True
                                                )
                                        swa = StochasticWeightAveraging(swa_lrs=1e-8,
                                                                        annealing_epochs=35,
                                                                        swa_epoch_start=185,
                                                                        )
                                        callbacks = [val_ckeckpoint, lr_monitor, early_stopping, swa, RichModelSummary()] #, DeviceStatsMonitor()
                                        profiler = PyTorchProfiler()
                                        trainer = L.Trainer(enable_checkpointing=True, max_epochs=250, accelerator="gpu", callbacks=callbacks, logger=logger) #precision="16-mixed", 
                                        #trainer = L.Trainer(enable_checkpointing=False, max_epochs=2, accelerator="cpu") #precision="16-mixed", 

                                        tuner = Tuner(trainer)
                                        lr_find_results = tuner.lr_find(model)
                                        fig = lr_find_results.plot(suggest=True)
                                        logger.experiment.add_figure("lr_find", fig)
                                        new_lr = lr_find_results.suggestion()
                                        model.hparams.lr = new_lr

                                        trainer.fit(model)
                                i += 1