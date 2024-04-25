import torch
import os
import math
import pandas as pd

parent_path = os.path.dirname(os.path.abspath(__file__))




# ==================== Latent Dimension Scaling ====================
model_dir = parent_path + "/../lightning_logs/AE_nPrune_02_nLayers_LatentScaling"
versions = [f.path for f in os.scandir(os.path.abspath(model_dir)) if f.is_dir()]

df = pd.DataFrame({'int': [], 'int': [], 'float': [], 'int': [], 'int': []}, columns = ['VersionID', 'best_epoch', 'val_loss', 'batch_size', 'latent_dim', 'n_layers'])
for i,version_path in enumerate(versions):
    path_i = version_path + "/checkpoints/last.ckpt"

    try:
        versionID = int(version_path[str(version_path).rfind('_')+1:])
        #print(f"version: {versionID}")
        checkpoint = torch.load(path_i,map_location=torch.device('cpu'))
        best_epoch = int(checkpoint["epoch"])
        validation_loss = math.inf
        for callback in checkpoint.get('callbacks', []):
            if isinstance(callback, str) and callback.startswith("ModelCheckpoint"):
                validation_loss = checkpoint["callbacks"][callback]["best_model_score"].item()
                break

        batch_size = checkpoint["hyper_parameters"]["batch_size"]
        n_layers = checkpoint["hyper_parameters"]["n_layers"]
        latent_dim = checkpoint["hyper_parameters"]["latent_dim"]
        print(versionID, best_epoch, validation_loss, batch_size, n_layers, latent_dim)
        df.loc[i] = {'VersionID': versionID, 'best_epoch': best_epoch, 'val_loss': validation_loss, 'batch_size': batch_size, 'latent_dim': latent_dim, 'n_layers': n_layers}

    except:
        print("skipping "+version_path)


    #row = pd.DataFrame({'int': [versionID], 'int': [best_epoch], 'float': [validation_loss], 'int': [batch_size], 'int': [n_layers]}, columns = ['VersionID', 'best_epoch', 'val_loss', 'batch_size', 'fc_dims'])
    #print(row)


df.to_csv('scan_nPrune_02_LatentScaling.csv', index=False)



# ==================== Layer Scaling ====================
model_dir = parent_path + "/../lightning_logs/AE_nPrune_02_nLayers"
versions = [f.path for f in os.scandir(os.path.abspath(model_dir)) if f.is_dir()]
df = pd.DataFrame({'int': [], 'int': [], 'float': [], 'int': [], 'int': []}, columns = ['VersionID', 'best_epoch', 'val_loss', 'batch_size', 'n_layers'])


for i,version_path in enumerate(versions):
    versionID = int(version_path[str(version_path).rfind('_')+1:])
    print(f"version: {versionID}")
    path_i = version_path + "/checkpoints/last.ckpt"
    checkpoint = torch.load(path_i,  map_location=torch.device('cpu'))
    best_epoch = int(checkpoint["epoch"])
    validation_loss = math.inf
    for callback in checkpoint.get('callbacks', []):
        if isinstance(callback, str) and callback.startswith("ModelCheckpoint"):
            validation_loss = checkpoint["callbacks"][callback]["best_model_score"].item()
            break

    batch_size = checkpoint["hyper_parameters"]["batch_size"]
    n_layers = checkpoint["hyper_parameters"]["n_layers"]
    latent_dim = checkpoint["hyper_parameters"]["latent_dim"]

    #row = pd.DataFrame({'int': [versionID], 'int': [best_epoch], 'float': [validation_loss], 'int': [batch_size], 'int': [n_layers]}, columns = ['VersionID', 'best_epoch', 'val_loss', 'batch_size', 'fc_dims'])
    #print(row)
    print(versionID, best_epoch, validation_loss, batch_size, n_layers, latent_dim)
    df.loc[i] = {'VersionID': versionID, 'best_epoch': best_epoch, 'val_loss': validation_loss, 'batch_size': batch_size, 'latent_dim': latent_dim, 'n_layers': n_layers}


df.to_csv('scan_nPrune_02_LayerScaling.csv', index=False)