import argparse
import json
import os
import shutil

import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm

import src.model as module_model
from src.preprocessing.melspec import MelSpectrogram
from src.utils import ROOT_PATH
from src.utils.parse_config import ConfigParser

DEVICE = "cuda:0" if torch.cuda.is_available() else 'cpu'


def get_data(target_sr):
    # 3 test utterances
    tests = [ 
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space",
    ]

    mel_transform = MelSpectrogram()

    data_list = []
    for i in range(1, 3 + 1):
        file_path = ROOT_PATH / 'data' / 'test_data' / f'Audio_{i}.wav'
        audio_tensor, sr = torchaudio.load(str(file_path))

        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first

        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        
        data_list.append(mel_transform(audio_tensor))

    return data_list

@torch.inference_mode()
def run_synthesis(model, sr):
    model.eval()
    data_list = get_data(sr)

    save_dir = ROOT_PATH / 'results'
    if save_dir.exists():
        shutil.rmtree(save_dir) # clean dir
    save_dir.mkdir(exist_ok=True, parents=True)

    for i, mel_spec in enumerate(data_list):
        generated_audio = model(mel_spec.to(DEVICE))['generated_audio'].squeeze(0)
        generated_audio = generated_audio.detach().cpu()
        torchaudio.save(str(save_dir / f't={i}.wav'), generated_audio, sr)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="Synthesize")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="path to checkpoint config file (default: None)",
    )
    args.add_argument(
        "-p",
        "--pretrained",
        default=None,
        type=str,
        help="path to latest checkpoint to init model weights with it (default: None)",
    )
    
    args = args.parse_args()
    
    assert args.config is not None
    assert args.pretrained is not None, 'Provide model checkpoint to use in script mode'

    with open(args.config, 'r') as f:
        config = ConfigParser(json.load(f))

    logger = config.get_logger("test")

    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    state_dict = torch.load(args.pretrained, map_location=DEVICE)['state_dict']
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    run_synthesis(model, sr=config["trainer"].get("sample_rate"))
