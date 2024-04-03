"""Simple fully connected model with pytorch lightning."""
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightning as L
import h5py
torch.set_float32_matmul_precision("high")
import numpy as np
from torch.func import functional_call, grad, vmap


class DiffLoss(nn.Module):
    def __init__(self, ylen: int, loss = nn.MSELoss(), eps = 1e-4):
        super().__init__()
        self.loss = loss
        
    def forward(self, inputs, pred, targets):   
        pred.backward(torch.ones_like(pred))
        ll = inputs.grad
        print(ll.shape)
        #l = self.loss(pred, targets)
        #l.backward()
        #ll = l.grad
        return self.loss(ll, targets)


class CustomDataset(Dataset):
    """Custom dataset class.
    Scales the data to have zero mean and unit variance (both x and y).
    Normalize each feature independently.
    If you want to use the data in the original scale, use the unnormalize_x and unnormalize_y methods.
    """
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x.clone().detach()#dtype=torch.float32)
        self.y = y.clone().detach()#dtype=torch.float32)
        self.ylen = y.shape[1] // 2

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
        x_norm = self.normalize_x(self.x[idx,:])
        y_norm = self.normalize_y(self.y[idx,:])
        return x_norm, y_norm

class SimpleFC_diff(L.LightningModule):
    def __init__(self, hparams: dict) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.batch_size = hparams["batch_size"]
        self.lr = hparams["lr"]
        self.dropout_in = nn.Dropout(hparams["dropout_in"]) if hparams["dropout_in"] > 0 else nn.Identity()
        self.dropout = nn.Dropout(hparams["dropout"]) if hparams["dropout"] > 0  else nn.Identity()
        self.activation_str = hparams["activation"]
        self.in_dim = hparams["in_dim"]
        self.loss_str = hparams["loss"] if "loss" in hparams else "MSE"
        ylen = 100


        self.loss_str = "diffLoss"
        self.loss = DiffLoss(ylen)

        self.activation = nn.ReLU()
        self.linear_layers = []

        self.out_dim = self.in_dim

        def linear_block(Nout, last_layer=False):
            res = [
                nn.Linear(self.out_dim, Nout),
                self.activation if (not last_layer) else nn.Identity() 
            ]
            self.out_dim = Nout
            return res
            
        def dense_layer(i):
            last_block = (i == len(hparams["fc_dims"]) - 1)

            if isinstance(hparams["fc_dims"][i], tuple):
                bl = []
                for j in range(hparams["fc_dims"][i][1]):
                    bl.extend(linear_block(hparams["fc_dims"][i][0], last_layer = last_block and (j == hparams["fc_dims"][i][1] - 1)))
                return nn.Sequential(*bl)
            else:
                return nn.Sequential(*linear_block(hparams["fc_dims"][i]))

        # append layers with dropout and norm
        for i in range(len(hparams["fc_dims"])):
            self.linear_layers.append(dense_layer(i))

        self.linear_layers = nn.ModuleList(self.linear_layers)
        #self.skip = nn.Linear(self.in_dim, self.out_dim)

        # initialize weights, might not be necessary
        for layer in self.linear_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(layer.bias)
        #nn.init.kaiming_normal_(self.skip.weight)
        #nn.init.zeros_(self.skip.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass
        model(x)
        """
        for layer in self.linear_layers:
            x = layer(x)
        return x

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step"""
        torch.set_grad_enabled(True)
        x, y = batch
        x.requires_grad = True
        y.requires_grad = True
        y_hat = self(x)  # self.forward(x)
        loss = self.loss(x, y_hat, y)
        self.log("train_loss", loss, prog_bar=False)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Validation step"""
        torch.set_grad_enabled(True)
        x, y = batch
        x.requires_grad = True
        y.requires_grad = True
        y_hat = self(x)
        loss = self.loss(x, y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Test step"""
        torch.set_grad_enabled(True)
        x, y = batch
        x.requires_grad = True
        y.requires_grad = True
        y_hat = self(x)
        loss = self.loss(x, y_hat, y)
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

    def setup(self, stage: str = None) -> None:
        """Called at the beginning of fit and test."""
        train_path = self.hparams["train_path"]
        with h5py.File(train_path, "r") as hf:
            x = hf["Set1/GImp"][:]
            y = hf["Set1/SImp"][:]
            ndens = hf["Set1/dens"][:]
        x = np.concatenate((x.real, x.imag), axis=1)
        y = np.concatenate((y.real, y.imag), axis=1)
        x = np.c_[ndens, x]
        p = np.random.RandomState(seed=0).permutation(x.shape[0])
        x = x[p,:]
        y = y[p,:]
        # convert from complex to two real numbers and then concatenate

        # split data using pytorch lightning
        x = torch.tensor(x, dtype=torch.float32) #dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32) #dtype=torch.float32)
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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        """Return val dataloader"""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader"""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8, persistent_workers=True)


