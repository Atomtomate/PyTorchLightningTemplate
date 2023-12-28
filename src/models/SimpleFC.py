"""Simple fully connected model with pytorch lightning."""
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightning as L
import h5py
from torchvision import transforms
torch.set_float32_matmul_precision("medium")

def h5_to_dataset(h5path: str) -> Dataset:
    """Read data from h5path and return a Dataset object.
    :param h5path: path to the data
    :return: Dataset object
    """
    with h5py.File(h5path, "r") as hf:
        x = hf["x"][:]
        y = hf["y"][:]

    dataset = CustomDataset(x, y)
    return dataset

class CustomDataset(Dataset):
    """Custom dataset class.
    Scales the data to have zero mean and unit variance (both x and y).
    Normalize each feature independently.
    If you want to use the data in the original scale, use the unnormalize_x and unnormalize_y methods.
    """
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.x_mean = torch.mean(self.x, dim=0)
        self.x_std = torch.std(self.x, dim=0)
        self.y_mean = torch.mean(self.y, dim=0)
        self.y_std = torch.std(self.y, dim=0)

    def __len__(self) -> int:
        return len(self.x)

    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        return x
        return (x - self.x_mean) / self.x_std

    def unnormalize_x(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.x_std + self.x_mean

    def normalize_y(self, x: torch.Tensor) -> torch.Tensor:
        return x
        return (x - self.y_mean) / self.y_std

    def unnormalize_y(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.y_std + self.y_mean

    def __getitem__(self, idx: int) -> tuple:
        x_norm = self.normalize_x(self.x[idx])
        y_norm = self.normalize_y(self.y[idx])
        return x_norm, y_norm

class SimpleFC_Lit(L.LightningModule):
    """Simple fully connected model with pytorch lightning."""
    def __init__(self, hparams: dict, verbose: bool=True) -> None:
        super().__init__()
        self.verbose = verbose
        self.save_hyperparameters()
        self.batch_size = hparams["batch_size"]
        self.lr = hparams["lr"]

        self.linear_layers = []
        self.linear_layers.append(nn.Linear(hparams["input_dim"], hparams["fc_dims"][0]))
        self.linear_layers.append(nn.BatchNorm1d(hparams["fc_dims"][0]))
        for i in range(len(hparams["fc_dims"])-1):
            self.linear_layers.append(nn.Linear(hparams["fc_dims"][i], hparams["fc_dims"][i+1]))
            self.linear_layers.append(nn.BatchNorm1d(hparams["fc_dims"][i+1]))
        self.linear_layers.append(nn.Linear(hparams["fc_dims"][-1], hparams["output_dim"]))
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.loss = nn.MSELoss()
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(hparams["dropout"])
        self.skip = nn.Linear(hparams["input_dim"], hparams["fc_dims"][-1])
        self.skip_bn = nn.BatchNorm1d(hparams["fc_dims"][-1])

        # initialize weights, might not be necessary
        for layer in self.linear_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="leaky_relu")
                nn.init.zeros_(layer.bias)
        nn.init.kaiming_normal_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # x_skip = self.skip(x)
        # x_skip = self.skip_bn(x_skip)
        # x_skip = self.dropout(x_skip)
        for layer in self.linear_layers[:-1]:
            x = self.activation(layer(x))
            # x = self.dropout(x)
        # x += x_skip
        x = self.linear_layers[-1](x)  # last layer has no activation
        return x

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step"""
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
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
        print("y_hat", y_hat[0:3])
        print("y", y[0:3])
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self) -> dict:
        """Configure optimizer"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def setup(self, stage: str) -> None:
        """Called at the beginning of fit and test."""
        if stage == "fit":
            self.train_dataset = h5_to_dataset(self.hparams["hparams"]["train_path"])
            self.val_dataset = h5_to_dataset(self.hparams["hparams"]["val_path"])
        elif stage == "test":
            self.test_dataset = h5_to_dataset(self.hparams["hparams"]["test_path"])


        if self.verbose:
            print("setup stage:", stage)
            if stage == "fit":
                print("train_dataset length:", len(self.train_dataset))
                print("val_dataset length:", len(self.val_dataset))
                print("x[0:3]", self.train_dataset[0:3][0])
                print("y[0:3]", self.train_dataset[0:3][1])
            elif stage == "test":
                print("test_dataset", len(self.test_dataset))
    def train_dataloader(self) -> DataLoader:
        """Return train dataloader"""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self) -> DataLoader:
        """Return val dataloader"""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader"""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)


