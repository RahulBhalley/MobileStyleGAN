import json
import torch
from MobileStyleGAN.core.utils import download_ckpt

def model_zoo(name, zoo_path="MobileStyleGAN/configs/model_zoo.json"):
    zoo = json.load(open(zoo_path))
    if name in zoo:
        ckpt = download_ckpt(**zoo[name])
    else:
        ckpt = torch.load(name, map_location="cpu")
    return ckpt
