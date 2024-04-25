from src.models.SimpleFC import SimpleFC_Lit
from src.models.FCSplit import FC_Split
from copy import deepcopy
import lightning as L
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, StochasticWeightAveraging, RichModelSummary, DeviceStatsMonitor
from lightning.pytorch.profilers import PyTorchProfiler
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import Profiler, PassThroughProfiler


torch.set_float32_matmul_precision("high")


hparams1 = {
            "model_name": "TMP_02",
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

#TODO: REDO 35
hparams_list = [hparams1]
i = 0
if __name__ == "__main__":
        for hparams in hparams_list:
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
                

                callbacks = [lr_monitor, RichModelSummary()] #, DeviceStatsMonitor()
                profiler = PyTorchProfiler(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA,],
                                           schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                                           dirpath="g:/Codes/PyTorchLightningTemplate/lightning_logs/log/tmp_02",
                                           filename="Prof",
                                           on_trace_ready=torch.profiler.tensorboard_trace_handler('g:/Codes/PyTorchLightningTemplate/lightning_logs/log/tmp_02'),
                                           record_shapes=True,
                                           profile_memory=True, 
                                           with_stack=True, 
                                           with_flops=True, 
                                           with_modules=True)
                trainer = L.Trainer(enable_checkpointing=False, 
                                    max_epochs=4, 
                                    precision="64-true", #"16-mixed", / 32/32-true
                                    devices=2, 
                                    accelerator="auto",
                                    callbacks=callbacks, logger=logger, profiler="simple") #precision="16-mixed", 
                #trainer = L.Trainer(enable_checkpointing=False, max_epochs=2, accelerator="cpu") #precision="16-mixed", 



                trainer.fit(model)