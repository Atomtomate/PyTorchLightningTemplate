import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightning as L
import h5py

dtype_default = torch.float64 #64

torch.set_float32_matmul_precision("highest")
torch.set_default_dtype(dtype_default)
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x.clone().detach().to(dtype=dtype_default)
        self.y = y.clone().detach().to(dtype=dtype_default)
        self.ylen = y.shape[1] // 2

    def __len__(self) -> int:
        return len(self.x)

    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def unnormalize_x(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def normalize_y(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def unnormalize_y(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def __getitem__(self, idx: int) -> tuple:
        x_norm = self.normalize_x(self.x[idx,:])
        y_norm = self.normalize_y(self.y[idx,:])
        return x_norm, y_norm

class AutoEncoder_01(L.LightningModule):
    def __init__(self, hparams: dict) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.batch_size = hparams["batch_size"]
        self.lr = hparams["lr"]
        self.dropout_in = nn.Dropout(hparams["dropout_in"]) if hparams["dropout_in"] > 0 else nn.Identity()
        self.dropout = nn.Dropout(hparams["dropout"]) if hparams["dropout"] > 0  else nn.Identity()
        self.activation_str = hparams["activation"]
        self.in_dim = hparams["in_dim"]
        self.latent_dim = hparams["latent_dim"]
        self.n_layers = hparams["n_layers"]
        self.loss_str = hparams["loss"] if "loss" in hparams else "MSE"
        ylen = 100


        if self.loss_str == "MSE":
            self.loss = nn.MSELoss()
        elif self.loss_str == "WeightedMSE":
            self.loss = WeightedLoss(ylen)
        elif self.loss_str == "WeightedMSE2":
            self.loss = WeightedLoss2(ylen)
        elif self.loss_str == "ScaledMSE":
            self.loss = ScaledLoss(ylen)
        elif self.loss_str == "WeightedScaledLoss":
            self.loss_str == WeightedScaledLoss(ylen)
        else:
            raise ValueError("unkown activation: " + self.hparams["activation"])

        if hparams["activation"] == "LeakyReLU":
            self.activation = nn.LeakyReLU() 
        elif hparams["activation"] == "SiLU":
            self.activation = nn.SiLU()
        elif hparams["activation"] == "ReLU":
            self.activation = nn.ReLU()
        else:
            raise ValueError("unkown activation: " + self.hparams["activation"])
        self.encoder = []
        self.decoder = []

        # tracks current out dim while building network layer-wise
        self.out_dim = self.in_dim

        def linear_block(Nout, i, last_layer=False):
            res = [
                self.dropout_in if i == 0 else self.dropout,
                nn.Linear(self.out_dim, Nout),
                nn.BatchNorm1d(Nout) if (not last_layer) and hparams["with_batchnorm"] else nn.Identity(),
                self.activation if (not last_layer) else nn.Identity() 
            ]
            return res
            
        ae_step_size =  (self.in_dim - self.latent_dim) // self.n_layers
        bl_encoder = []
        for i in range(self.n_layers):
            if i == self.n_layers - 1:
                bl_encoder.extend(linear_block(self.latent_dim, i, last_layer = True))
                self.out_dim = self.latent_dim
            else:
                bl_encoder.extend(linear_block(self.out_dim - ae_step_size, i))
                self.out_dim = self.out_dim - ae_step_size

        bl_decoder = []
        for i in range(self.n_layers):
            if i == self.n_layers - 1:
                bl_decoder.extend(linear_block(self.in_dim, i, last_layer = True))
                self.out_dim = self.in_dim
            else:
                bl_decoder.extend(linear_block(self.out_dim + ae_step_size, i))
                self.out_dim = self.out_dim + ae_step_size

        self.encoder = nn.Sequential(*bl_encoder) #nn.ModuleList()
        self.decoder = nn.Sequential(*bl_decoder) #nn.ModuleList()

        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(layer.bias)
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(layer.bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)  # self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=False)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self) -> dict:
        if self.hparams["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr,
                                    momentum=self.hparams["SGD_momentum"],
                                    weight_decay=self.hparams["SGD_weight_decay"],
                                    dampening=self.hparams["SGD_dampening"],
                                    nesterov=self.hparams["SGD_nesterov"])
        elif self.hparams["optimizer"] == "RMSprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr,
                        momentum=self.hparams["SGD_momentum"],
                        weight_decay=self.hparams["SGD_weight_decay"],
                        alpha=self.hparams["RMSprop_alpha"])
        
        elif self.hparams["optimizer"] == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        elif self.hparams["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            raise ValueError("unkown optimzer: " + self.hparams["optimizer"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def setup(self, stage: str = None) -> None:
        """Called at the beginning of fit and test."""
        train_path = self.hparams["train_path"]
        with h5py.File(train_path, "r") as hf:
            x = hf["Set1/GImp"][:]
            #y = hf["Set1/GImp"][:]
        x = np.concatenate((x.real, x.imag), axis=1)
        y = copy.deepcopy(x)
        p = np.random.RandomState(seed=0).permutation(x.shape[0])
        x = x[p,:]
        y = y[p,:]
        # convert from complex to two real numbers and then concatenate

        # split data using pytorch lightning
        x = torch.tensor(x, dtype=dtype_default)
        y = torch.tensor(y, dtype=dtype_default)
        n = x.shape[0]
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        n_test = n - n_train - n_val
        # shuffle data
        x_train = x[:n_train,:]
        y_train = y[:n_train,:]
        x_val = x[n_train:n_train+n_val,:]
        y_val = y[n_train:n_train+n_val,:]
        x_test = x[n_train+n_val:,:]
        y_test = y[n_train+n_val:,:]

        self.train_dataset = CustomDataset(x_train, y_train)
        self.val_dataset = CustomDataset(x_val, y_val)
        self.test_dataset = CustomDataset(x_test, y_test)

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader"""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        """Return val dataloader"""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8, persistent_workers=True, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader"""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8, persistent_workers=True, pin_memory=True)