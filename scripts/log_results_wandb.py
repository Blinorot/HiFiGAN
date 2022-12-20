import argparse
from pathlib import Path

import torchaudio
import wandb


def log(args):
    data_dir = Path(__file__).absolute().resolve().parent.parent
    data_dir = data_dir / "results"

    assert data_dir.exists(), "run synthesis first"

    with wandb.init(project=args.project, name=args.name):
        for fpath in data_dir.iterdir():
            audio, sr = torchaudio.load(fpath)
            index = fpath.name.split('=')[1][0]
            if index == '0':
                text = 'Defibrillator'
            elif index == '1':
                text = 'Massachusetts'
            elif index == '2':
                text = 'Wasserstein'
            new_name = text
            audio = audio.detach().cpu().numpy().T
            wandb.log({
                new_name: wandb.Audio(audio, sample_rate=sr)
            })


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="Log results to wandb")
    args.add_argument(
        "-p",
        "--project",
        default="hifigan_project",
        type=str,
        help="WandB project name",
    )

    args.add_argument(
        "-n",
        "--name",
        default="results_log",
        type=str,
        help="WandB run name",
    )

    args = args.parse_args()

    log(args)
