from src.models.AE import AutoEncoder_01
from copy import deepcopy
import lightning as L
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, StochasticWeightAveraging, GradientAccumulationScheduler, RichModelSummary, DeviceStatsMonitor
from lightning.pytorch.profilers import PyTorchProfiler
import torch
from pytorch_lightning.loggers import TensorBoardLogger
#from ray import tune
#from ray.tune.schedulers import ASHAScheduler
#from hyperopt import hp
#from ray.tune.suggest.hyperopt import HyperOptSearch
import optuna
from optuna.integration import PyTorchLightningPruningCallback

dtype_default = torch.float32

torch.set_float32_matmul_precision("high")
torch.set_default_dtype(dtype_default)

hparams = {
            "model_name": "AE_nPrune_02_nLayers_SGD",
            "in_dim" : 200,
            "latent_dim": 10,
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

study = optuna.create_study()


def objective(trial):
        max_epochs = 50
        optimizer = "Adam" # trial.suggest_categorical("optimizer", ["SGD", "Adam"])
        lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
        SGD_weight_decay = 0.0 #trial.suggest_float("SGD_weight_decay", 0.0, 1.0)
        SGD_momentum = 0.0 #trial.suggest_float("SGD_momentum", 0.0, 1.0)
        SGD_dampening = 0.0 #trial.suggest_float("SGD_dampening", 0.0, 1.0)
        batch_size = trial.suggest_int("batch_size", 10, 1024, step=5)
        SGD_nesterov = False #trial.suggest_categorical("SGD_nesterov", [True, False])
        with_batchnorm = False # trial.suggest_categorical("with_batchnorm", [True, False])
        SGD_dampening = 0.0 #trial.suggest_float("SGD_dampening", 0.8, 1.0)
        RMSprop_alpha = 0.99 #trial.suggest_float("RMSprop_alpha", 0.8, 1.0)
        hparams["optimizer"] = optimizer
        hparams["dropout_in"] = 0.0
        hparams["dropout"] = 0.0
        hparams["lr"] = lr
        hparams["SGD_weight_decay"] = SGD_weight_decay
        hparams["SGD_momentum"] = SGD_momentum
        hparams["SGD_dampening"] = SGD_dampening
        hparams["batch_size"] = batch_size
        hparams["SGD_nesterov"] = SGD_nesterov
        hparams["with_batchnorm"] = with_batchnorm
        hparams["RMSprop_alpha"] = RMSprop_alpha

                                
        model = AutoEncoder_01(hparams) 
        early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=10)
        logger = TensorBoardLogger("lightning_logs", name=hparams["model_name"])
        swa = StochasticWeightAveraging(swa_lrs=1e-8,
                                        annealing_epochs=10,
                                        swa_epoch_start=80,
                                        )
        #accumulator = GradientAccumulationScheduler(scheduling={0: 256, 4: 128, 10: 64, 18: 32, 26: 16, 34: 8, 42: 4, 50: 1})
        pruning = PyTorchLightningPruningCallback(trial, monitor="val_loss")
        callbacks = [early_stopping, pruning] #, accumulator, DeviceStatsMonitor()
        trainer = L.Trainer(enable_checkpointing=False, 
                            max_epochs=max_epochs, 
                            accelerator="cpu", 
                            callbacks=callbacks,
                            logger=logger) #precision="16-mixed", 

        trainer.logger.log_hyperparams(hparams)
        trainer.fit(model)
        return trainer.callback_metrics["val_loss"].item()

if __name__ == "__main__":
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(direction="minimize", pruner=pruner,  
                                        storage="sqlite:///db.sqlite3", 
                                        study_name="3_layer_Adam_lr_bs",
                                        load_if_exists=True)
    study.optimize(objective, n_trials=500, gc_after_trial=True)
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
