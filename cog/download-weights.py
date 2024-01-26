#!/usr/bin/env python

import os
import sys
import torch
import time
import subprocess
from diffusers import StableDiffusionXLPipeline

# append project directory to path so predict.py can be imported
sys.path.append('.')

from predict import SD_MODEL_NAME, SD_MODEL_CACHE

# for ip-adapter and ControlNetModel
CHECKPOINTS_CACHE = "./checkpoints"
CHECKPOINT_IP_ADAPTER_URL = "https://huggingface.co/InstantX/InstantID/blob/main/ip-adapter.bin"
CHECKPOINT_CTRLNET_URL = "https://huggingface.co/InstantX/InstantID/blob/main/ControlNetModel/diffusion_pytorch_model.safetensors"
CHECKPOINT_CTRLNET_CFG_URL = "https://huggingface.co/InstantX/InstantID/blob/main/ControlNetModel/config.json"

# for `models/antelopev2`
MODELS_CACHE = "./models"
MODELS_URL = "https://weights.replicate.delivery/default/InstantID/models.tar"

# Make cache folder
if not os.path.exists(SD_MODEL_CACHE):
    os.makedirs(SD_MODEL_CACHE)

# Download and save the SD model weights
pipe = StableDiffusionXLPipeline.from_pretrained(
  SD_MODEL_NAME,
  torch_dtype=torch.float16,
  safety_checker=None,
)
pipe.save_pretrained(SD_MODEL_CACHE)

# Download the ip-adapter and ControlNetModel checkpoints
def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

if not os.path.exists(MODELS_CACHE):
    download_weights(MODELS_URL, MODELS_CACHE)

if not os.path.exists(CHECKPOINTS_CACHE):
    download_weights(CHECKPOINT_IP_ADAPTER_URL, CHECKPOINTS_CACHE)
    download_weights(CHECKPOINT_CTRLNET_URL, f"{CHECKPOINTS_CACHE}\ControlNetModel")
    download_weights(CHECKPOINT_CTRLNET_CFG_URL, f"{CHECKPOINTS_CACHE}\ControlNetModel")
