from src.models.AE import AutoEncoder_01
from copy import deepcopy
import lightning as L
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, StochasticWeightAveraging, GradientAccumulationScheduler, RichModelSummary, DeviceStatsMonitor
from lightning.pytorch.profilers import PyTorchProfiler
import torch
from pytorch_lightning.loggers import TensorBoardLogger
torch.set_float32_matmul_precision("highest")
torch.set_default_dtype(torch.float64)

hparams0 = {
            "model_name": "AE_nPrune_02_nLayers_LayerScaling",
            "in_dim" : 200,
            "latent_dim": 30,
            "n_layers": 3,
            "dropout_in": 0.0,
            "dropout": 0.0,
            "with_batchnorm": False,
            "lr": 0.01,
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
                for bs in [256, 64]:
                        hparams["batch_size"] = bs
                        for n_layers in range(4,10,1): #[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
                                hparams["n_layers"] = n_layers
                                for hl in [8]:
                                        if True:
                                                max_epochs = 10 if hl > 18 else 300
                                                hparams["latent_dim"] = hl
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
                                                                                swa_epoch_start=250,
                                                                                )
                                                accumulator = GradientAccumulationScheduler(scheduling={0: 32, 8: 16, 16: 8, 24: 4, 32: 1})
                                                callbacks = [val_ckeckpoint, lr_monitor, early_stopping, swa, RichModelSummary(), accumulator] #, DeviceStatsMonitor()
                                                profiler = PyTorchProfiler()
                                                trainer = L.Trainer(enable_checkpointing=True, max_epochs=max_epochs, accelerator="gpu", callbacks=callbacks, logger=logger) #precision="16-mixed", 
 

                                                tuner = Tuner(trainer)
                                                lr_find_results = tuner.lr_find(model, min_lr=1e-09, num_training=200)
                                                fig = lr_find_results.plot(suggest=True)
                                                logger.experiment.add_figure("lr_find", fig)
                                                new_lr = lr_find_results.suggestion()
                                                model.hparams.lr = new_lr

                                                trainer.fit(model)
                                        i += 1