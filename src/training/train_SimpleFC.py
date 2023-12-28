from src.models.SimpleFC import SimpleFC_Lit
import lightning as L
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, StochasticWeightAveraging
import torch
torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    hparams = {"input_dim": 2,
            "fc_dims": [50, 50],
            "dropout": 0.4,
            "output_dim": 2,
            "lr": 0.01,
            "batch_size": 128,
            "train_path": "runs/example_run/data/data_train.h5",
            "val_path": "runs/example_run/data/data_val.h5",
            "test_path": "runs/example_run/data/data_test.h5",
               }
    model = SimpleFC_Lit(hparams)
    val_ckeckpoint = ModelCheckpoint( # saved in `trainer.default_root_dir`/`logger.version`/`checkpoint_callback.dirpath`
            filename="{epoch}-{step}-{val_loss:.5f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=50)
    swa = StochasticWeightAveraging(swa_lrs=0.0001,
                                    swa_epoch_start=50,
                                    )
    callbacks = [val_ckeckpoint, lr_monitor, early_stopping, swa]
    trainer = L.Trainer(max_epochs=100, accelerator="gpu", precision="32", callbacks=callbacks)
    # tuner = Tuner(trainer)
    # lr_finder = tuner.lr_find(model, num_training=1000)
    # model.hparams.lr = lr_finder.suggestion()
    

    trainer.fit(model)
    trainer.test(model)

