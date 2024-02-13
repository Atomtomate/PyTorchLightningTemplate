"""Simple fully connected model with pytorch lightning."""
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightning as L
import h5py
torch.set_float32_matmul_precision("high")
import numpy as np

class SplitDataset(Dataset):
    """Custom dataset class.
    Scales the data to have zero mean and unit variance (both x and y).
    Normalize each feature independently.
    If you want to use the data in the original scale, use the unnormalize_x and unnormalize_y methods.
    """
    def __init__(self, x: torch.Tensor, y_re: torch.Tensor, y_im: torch.Tensor) -> None:
        self.x = x.clone().detach()#dtype=torch.float32)
        self.y_re = y_re.clone().detach()#dtype=torch.float32)
        self.y_im = y_im.clone().detach()

    def __len__(self) -> int:
        return len(self.x)

    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        return x
        # return (x - self.x_mean) / self.x_std

    def unnormalize_x(self, x: torch.Tensor) -> torch.Tensor:
        return x
        #return x * self.x_std + self.x_mean

    def normalize_y(self, x: torch.Tensor) -> torch.Tensor:
        return x
        #return (x - self.y_mean) / self.y_std

    def unnormalize_y(self, x: torch.Tensor) -> torch.Tensor:
        return x
        #return x * self.y_std + self.y_mean

    def __getitem__(self, idx: int) -> tuple:
        #x_norm = self.normalize_x(self.x[idx])
        #y_norm = self.normalize_y(self.y_re[idx])
        return self.x[idx,:], self.y_im[idx,:] #, self.y_im[idx]

class FC_Split(L.LightningModule):
    """Simple fully connected model with pytorch lightning."""
    def __init__(self, hparams: dict, verbose: bool=True) -> None:
        super().__init__()
        self.verbose = verbose
        self.save_hyperparameters(hparams)
        self.batch_size = hparams["batch_size"]
        self.lr = hparams["lr"]
        self.dropout_in = nn.Dropout(hparams["dropout_in"]) if hparams["dropout_in"] > 0 else nn.Identity()
        self.dropout = nn.Dropout(hparams["dropout"]) if hparams["dropout"] > 0  else nn.Identity()
        self.activation = nn.SiLU() #nn.LeakyReLU() 
        self.linear_layers_re = []
        self.linear_layers_im = []

        def block(i):
            return nn.Sequential(
                self.dropout_in if i == 0 else self.dropout,
                nn.Linear(hparams["fc_dims"][i], hparams["fc_dims"][i+1]),
                nn.BatchNorm1d(hparams["fc_dims"][i+1]) if hparams["with_batchnorm"] else nn.Identity(),
                self.activation
            )
        
        # ===== Real Part =====
        # append layers with dropout and norm
        for i in range(len(hparams["fc_dims"])-1):
            self.linear_layers_re.append(block(i))
        # append output layer
        self.linear_layers_re.append(nn.Sequential(
            self.dropout,
            nn.Linear(hparams["fc_dims"][-2], hparams["fc_dims"][-1]))
        )
        self.linear_layers_re = nn.ModuleList(self.linear_layers_re)

        # ===== Imaginary Part =====
        # append layers with dropout and norm
        for i in range(len(hparams["fc_dims"])-1):
            self.linear_layers_im.append(block(i))
        # append output layer
        self.linear_layers_im.append(nn.Sequential(
            self.dropout,
            nn.Linear(hparams["fc_dims"][-2], hparams["fc_dims"][-1]))
        )
        self.linear_layers_im = nn.ModuleList(self.linear_layers_im)

        self.loss = nn.MSELoss() #weighted_MSELoss(hparams["fc_dims"][-1] // 2) #nn.MSELoss() #weighted_MSELoss() #
        self.skip = nn.Linear(hparams["fc_dims"][0], hparams["fc_dims"][-1])
        # initialize weights, might not be necessary
        #for layer in self.linear_layers_re:
        #    if isinstance(layer, nn.Linear):
        #        nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="leaky_relu")
        #        nn.init.zeros_(layer.bias)
        #nn.init.kaiming_normal_(self.skip.weight)
        #nn.init.zeros_(self.skip.bias)


    def forward(self, x: torch.Tensor):
        """Forward pass
        model(x)
        """
        #for layer in self.linear_layers_re:
        #    x_re = layer(x_re)
        for layer in self.linear_layers_re:
            x = layer(x)
        return x

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step"""
        x, y = batch
        y_hat = self(x)  # self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=False)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Validation step"""
        x, y = batch
        y_hat = self(x)  # self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Test step"""
        x, y = batch
        y_hat = self(x)  # self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self) -> dict:
        """Configure optimizer"""
        if self.hparams["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr,
                                    momentum=self.hparams["SGD_momentum"],
                                    weight_decay=self.hparams["SGD_weight_decay"],
                                    dampening=self.hparams["SGD_dampening"],
                                    nesterov=self.hparams["SGD_nesterov"])
        
        elif self.hparams["optimizer"] == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        elif self.hparams["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            raise ValueError("unkown optimzer: " + self.hparams["optimzer"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def setup(self, stage: str = None, dtype = torch.float32) -> None:
        """Called at the beginning of fit and test."""
        train_path = self.hparams["train_path"]
        with h5py.File(train_path, "r") as hf:
            x = hf["Set1/GImp"][:]
            y = hf["Set1/SImp"][:]
            ndens = hf["Set1/dens"][:]
        x = np.concatenate((x.real, x.imag), axis=1)
        y_re = y.real 
        y_im = y.imag 
        x = np.c_[ndens, x]
        p = np.random.RandomState(seed=0).permutation(x.shape[0])
        x = x[p,:]
        y_re = y_re[p,:]
        y_im = y_im[p,:]

        # split data
        x = torch.tensor(x, dtype=dtype)
        y_re = torch.tensor(y_re, dtype=dtype) 
        y_im = torch.tensor(y_im, dtype=dtype) 
        n = x.shape[0]
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        n_test = n - n_train - n_val
        # shuffle data
        x_train = x[:n_train,:]
        x_val = x[n_train:n_train+n_val,:]
        x_test = x[n_train+n_val:,:]

        y_re_train = y_re[:n_train,:]
        y_re_val = y_re[n_train:n_train+n_val,:]
        y_re_test = y_re[n_train+n_val:,:]
        y_im_train = y_im[:n_train,:]
        y_im_val = y_im[n_train:n_train+n_val,:]
        y_im_test = y_im[n_train+n_val:,:]

        self.train_dataset = SplitDataset(x_train, y_re_train, y_im_train)
        self.val_dataset = SplitDataset(x_val, y_re_val, y_im_val)
        self.test_dataset = SplitDataset(x_test, y_re_test, y_im_test)

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader"""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        """Return val dataloader"""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader"""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8, persistent_workers=True)


