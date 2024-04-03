import torch
import os
import math
import pandas as pd

parent_path = os.path.dirname(os.path.abspath(__file__))
model_dir = parent_path + "/../lightning_logs/FullCN_nPrune_02_nLayers"
versions = [f.path for f in os.scandir(os.path.abspath(model_dir)) if f.is_dir()]

df = pd.DataFrame({'int': [], 'int': [], 'float': [], 'int': [], 'int': []}, columns = ['VersionID', 'best_epoch', 'val_loss', 'batch_size', 'fc_dims'])


for i,version_path in enumerate(versions):
    versionID = int(version_path[str(version_path).rfind('_')+1:])
    print(f"version: {versionID}")
    path_i = version_path + "/checkpoints/last.ckpt"
    checkpoint = torch.load(path_i)
    best_epoch = int(checkpoint["epoch"])
    validation_loss = math.inf
    for callback in checkpoint.get('callbacks', []):
        if isinstance(callback, str) and callback.startswith("ModelCheckpoint"):
            validation_loss = checkpoint["callbacks"][callback]["best_model_score"].item()
            break

    batch_size = checkpoint["hyper_parameters"]["batch_size"]
    n_layers = checkpoint["hyper_parameters"]["fc_dims"][0][1]

    #row = pd.DataFrame({'int': [versionID], 'int': [best_epoch], 'float': [validation_loss], 'int': [batch_size], 'int': [n_layers]}, columns = ['VersionID', 'best_epoch', 'val_loss', 'batch_size', 'fc_dims'])
    #print(row)
    print(versionID, best_epoch, validation_loss, batch_size, n_layers)
    df.loc[i] = {'VersionID': versionID, 'best_epoch': best_epoch, 'val_loss': validation_loss, 'batch_size': batch_size, 'fc_dims': n_layers}


df.to_csv('scan_nPrune_02_nLayers.csv', index=False)