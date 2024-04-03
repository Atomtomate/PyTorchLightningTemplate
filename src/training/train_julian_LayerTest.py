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


hparams1 = {
            "model_name": "FullCN_nPrune_02_nLayers",
            "in_dim" : 201,
            "fc_dims": [(200,3)],
            "dropout_in": 0.0,
            "dropout": 0.0,
            "with_batchnorm": False,
            "lr": 0.001584893192461114, #0.01,
            "batch_size": 128,
            "train_path": "D:/data_batch2_nPrune.hdf5",
            "optimizer": "Adam",
            "activation": "ReLU",
            "SGD_weight_decay": 0.0,
            "SGD_momentum": 0.9,
            "SGD_dampening": 0.0,
            "SGD_nesterov": False,
            "loss": "MSE",
        }

hparams2 = deepcopy(hparams1)
hparams2["fc_dims"] = [(200,4)]
hparams3 = deepcopy(hparams1)
hparams3["fc_dims"] = [(200,5)]
hparams4 = deepcopy(hparams1)
hparams4["fc_dims"] = [(200,6)]
hparams5 = deepcopy(hparams1)
hparams5["fc_dims"] = [(200,7)]
hparams6 = deepcopy(hparams1)
hparams6["fc_dims"] = [(200,8)]
hparams7 = deepcopy(hparams1)
hparams7["fc_dims"] = [(200,9)]
hparams8 = deepcopy(hparams1)
hparams8["fc_dims"] = [(200,10)]
#TODO: REDO 35
hparams_list = [hparams1, hparams2, hparams3, hparams4, hparams5, hparams6, hparams7, hparams8]
i = 0
if __name__ == "__main__":
    for bs in [4096, 2048, 1024, 256, 128, 64, 32, 16]: 
        for hparams in hparams_list:
                if i > 41:
                        hparams["batch_size"] = bs
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
                                                        swa_epoch_start=50,
                                                        )
                        callbacks = [val_ckeckpoint, lr_monitor, early_stopping, swa, RichModelSummary()] #, DeviceStatsMonitor()
                        profiler = PyTorchProfiler()
                        trainer = L.Trainer(enable_checkpointing=True, max_epochs=150, accelerator="gpu", callbacks=callbacks, logger=logger) #precision="16-mixed", 
                        #trainer = L.Trainer(enable_checkpointing=False, max_epochs=2, accelerator="cpu") #precision="16-mixed", 

                        tuner = Tuner(trainer)
                        lr_find_results = tuner.lr_find(model)
                        fig = lr_find_results.plot(suggest=True)
                        logger.experiment.add_figure("lr_find", fig)
                        new_lr = lr_find_results.suggestion()
                        model.hparams.lr = new_lr

                        trainer.fit(model)
                i += 1