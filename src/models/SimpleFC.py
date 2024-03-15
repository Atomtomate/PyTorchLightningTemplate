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

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)

def compute_jacobian(inputs, output):
    """
    :param inputs: Batch X Size (e.g. Depth X Width X Height)
    :param output: Batch X Classes
    :return: jacobian: Batch X Classes X Size
    From: https://discuss.pytorch.org/t/implementing-jacobian-differential-for-loss-function/35815
    """
    assert inputs.requires_grad

    num_classes = output.size()[1]

    jacobian = torch.zeros(num_classes, *inputs.size())
    grad_output = torch.zeros(*output.size())
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    for i in range(num_classes):
        zero_gradients(inputs)
        grad_output.zero_()
        grad_output[:, i] = 1
        output.backward(grad_output, retain_graph=True)
        jacobian[i] = inputs.grad.data

    return torch.transpose(jacobian, dim0=0, dim1=1)

class DiffLoss(nn.Module):
    def __init__(self, ylen: int, loss = nn.MSELoss(), eps = 1e-4):
        super().__init__()
        
    def forward(self,pred,targets):    
        dfdx = vmap(grad(f), in_dims=(0, None))

class ScaledLoss(nn.Module):
    def __init__(self, ylen: int, loss = nn.MSELoss(), eps = 1e-4):
        super().__init__()
        self.ylen = ylen
        self.dist = nn.PairwiseDistance(p=2, keepdim = True)
        self.loss = loss
        self.eps  = eps

    def forward(self,pred,targets):
        dist_re = torch.clamp(torch.max(targets[:,:self.ylen],dim=1,keepdim=True).values -
                    torch.min(targets[:,:self.ylen],dim=1,keepdim=True).values, min=self.eps, max=self.eps)
        dist_im = torch.clamp(torch.max(targets[:,self.ylen:],dim=1,keepdim=True).values - 
                    torch.min(targets[:,self.ylen:],dim=1,keepdim=True).values, min=self.eps, max=self.eps)
        loss_re = self.loss(pred[:,:self.ylen] / dist_re, targets[:,:self.ylen] / dist_re)
        loss_im = self.loss(pred[:,self.ylen:] / dist_im, targets[:,self.ylen:] / dist_im)
        return loss_re + loss_im

class WeightedLoss(nn.Module):
    def __init__(self, ylen: int, loss = nn.MSELoss()):
        super().__init__()
        self.ylen = ylen
        self.dist = nn.PairwiseDistance(p=2, keepdim = True)
        self.loss = loss

    def forward(self,pred,targets):

        dist_re = self.dist(
                        torch.max(targets[:,:self.ylen],dim=1,keepdim=True).values, 
                        torch.min(targets[:,:self.ylen],dim=1,keepdim=True).values)
        dist_im = self.dist(
                        torch.max(targets[:,self.ylen:],dim=1,keepdim=True).values, 
                        torch.min(targets[:,self.ylen:],dim=1,keepdim=True).values)
        scale_re = dist_im / (dist_re + dist_im)
        scale_im = dist_re / (dist_re + dist_im)
        loss_re = self.loss(scale_re * pred[:,:self.ylen], scale_re * targets[:,:self.ylen])
        loss_im = self.loss(scale_im * pred[:,self.ylen:], scale_im * targets[:,self.ylen:])
        return loss_re + loss_im
    
class WeightedLoss2(nn.Module):
    def __init__(self, ylen: int, loss = nn.MSELoss()):
        super().__init__()
        self.ylen = ylen
        self.dist = nn.PairwiseDistance(p=2, keepdim = True)
        self.loss = loss

    def forward(self,pred,targets):

        dist_re = self.dist(
                        torch.max(targets[:,:self.ylen],dim=1,keepdim=True).values, 
                        torch.min(targets[:,:self.ylen],dim=1,keepdim=True).values)
        dist_im = self.dist(
                        torch.max(targets[:,self.ylen:],dim=1,keepdim=True).values, 
                        torch.min(targets[:,self.ylen:],dim=1,keepdim=True).values)
        scale_re = dist_re / (dist_re + dist_im)
        scale_im = dist_im / (dist_re + dist_im)
        loss_re = self.loss(scale_re * pred[:,:self.ylen], scale_re * targets[:,:self.ylen])
        loss_im = self.loss(scale_im * pred[:,self.ylen:], scale_im * targets[:,self.ylen:])
        return loss_re + loss_im
    
class WeightedScaledLoss(nn.Module):
    def __init__(self, ylen: int, loss = nn.MSELoss(), eps = 1e-4):
        super().__init__()
        self.ylen = ylen
        self.dist = nn.PairwiseDistance(p=2, keepdim = True)
        self.loss = loss
        self.eps  = eps

    def forward(self,pred,targets):

        dist_re = torch.clamp(torch.max(targets[:,:self.ylen],dim=1,keepdim=True).values -
                    torch.min(targets[:,:self.ylen],dim=1,keepdim=True).values, min=self.eps, max=self.eps)
        dist_im = torch.clamp(torch.max(targets[:,self.ylen:],dim=1,keepdim=True).values - 
                    torch.min(targets[:,self.ylen:],dim=1,keepdim=True).values, min=self.eps, max=self.eps)
        scale_re = dist_im / (dist_re*(dist_re + dist_im))
        scale_im = dist_re / (dist_im*(dist_re + dist_im))
        loss_re = self.loss(scale_re * pred[:,:self.ylen], scale_re * targets[:,:self.ylen])
        loss_im = self.loss(scale_im * pred[:,self.ylen:], scale_im * targets[:,self.ylen:])
        return loss_re + loss_im

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

class SimpleFC_Lit(L.LightningModule):
    """Simple fully connected model with pytorch lightning."""
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
        self.linear_layers = []

        self.out_dim = self.in_dim

        def linear_block(Nout, last_layer=False):
            res = [
                self.dropout_in if i == 0 else self.dropout,
                nn.Linear(self.out_dim, Nout),
                nn.BatchNorm1d(Nout) if (not last_layer) and hparams["with_batchnorm"] else nn.Identity(),
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
        self.skip = nn.Linear(self.in_dim, self.out_dim)

        # initialize weights, might not be necessary
        for layer in self.linear_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="linear")
                nn.init.zeros_(layer.bias)
        nn.init.kaiming_normal_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass
        model(x)
        """
        for layer in self.linear_layers:
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
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Test step"""
        x, y = batch
        y_hat = self(x)
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


