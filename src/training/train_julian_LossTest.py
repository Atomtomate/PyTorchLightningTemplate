from src.models.SimpleFC import SimpleFC_Lit
from src.models.FCSplit import FC_Split
from copy import deepcopy
import lightning as L
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, StochasticWeightAveraging, RichModelSummary, DeviceStatsMonitor
from lightning.pytorch.profilers import PyTorchProfiler
import torch
from pytorch_lightning.loggers import TensorBoardLogger
torch.set_float32_matmul_precision("high")



hparams0 = {
            "model_name": "FullCN_LossTest",
            "in_dim" : 201,
            "fc_dims": [(200,10)],
            "dropout_in": 0.0,
            "dropout": 0.0,
            "with_batchnorm": False,
            "lr": 0.001584893192461114, #0.01,
            "batch_size": 512,
            "train_path": "D:/data_test2.hdf5",
            "optimizer": "Adam",
            "activation": "ReLU",
            "SGD_weight_decay": 0.0,
            "SGD_momentum": 0.9,
            "SGD_dampening": 0.0,
            "SGD_nesterov": False,
            "loss": "MSE",
        }

hparams1 = deepcopy(hparams0)
hparams1["loss"] = "WeightedMSE"
hparams2 = deepcopy(hparams0)
hparams2["loss"] = "WeightedMSE2"
hparams3 = deepcopy(hparams0)
hparams3["loss"] = "ScaledMSE"
hparams4 = deepcopy(hparams0)
hparams4["loss"] = "WeightedScaledLoss"

hparams_list = [hparams0, hparams1, hparams2, hparams3, hparams4]
i = 0
if __name__ == "__main__":
        for hparams in hparams_list:
                if i > 0:
                        model = SimpleFC_Lit(hparams) #FC_Split(hparams)
                        lr_monitor = LearningRateMonitor(logging_interval='step')
                        early_stopping = EarlyStopping(
                                monitor="val_loss",
                                patience=50)
                        logger = TensorBoardLogger("lightning_logs", name=hparams["model_name"])
                        val_ckeckpoint = ModelCheckpoint( # saved in `trainer.default_root_dir`/`logger.version`/`checkpoint_callback.dirpath`
                                filename="{epoch}-{step}-{val_loss:.8f}",
                                monitor="val_loss",
                                mode="min",
                                save_top_k=10,
                                save_last =True
                                )
                        swa = StochasticWeightAveraging(swa_lrs=0.0001,
                                                        swa_epoch_start=100,
                                                        )
                        callbacks = [val_ckeckpoint, lr_monitor, early_stopping, swa, RichModelSummary()] #, DeviceStatsMonitor()
                        profiler = PyTorchProfiler()
                        trainer = L.Trainer(enable_checkpointing=True, max_epochs=200, accelerator="gpu", callbacks=callbacks, logger=logger) #precision="16-mixed", 
                        #trainer = L.Trainer(enable_checkpointing=False, max_epochs=2, accelerator="cpu") #precision="16-mixed", 

                        tuner = Tuner(trainer)
                        lr_find_results = tuner.lr_find(model)
                        fig = lr_find_results.plot(suggest=True)
                        logger.experiment.add_figure("lr_find", fig)
                        new_lr = lr_find_results.suggestion()
                        model.hparams.lr = new_lr

                        trainer.fit(model)
                i += 1