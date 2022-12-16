# HiFiGAN

This is a repository for Vocoder homework of HSE DLA Course. The implemented model is HiFiGAN trained on LJSpeech.

## Getting Started

These instructions will help you to run a project on your machine.

### Prerequisites

Install [pyenv](https://github.com/pyenv/pyenv#installation) following the instructions in the official repository.

### Installation

Clone the repository into your local folder:

```bash
cd path/to/local/folder
git clone https://github.com/Blinorot/HiFiGAN.git
```

Install `python 3.9.7.` for `pyenv` and create virtual environment:

```bash
pyenv install 3.9.7
cd path/to/cloned/HiFiGAN/project
~/.pyenv/versions/3.9.7/bin/python -m venv hf_env
```

Install required python packages to python environment:

```bash
source hf_env/bin/activate
pip install -r requirements.txt
```

### Data downloading and preprocessing

To download data run the following command:

```bash
python3 scripts/download_data.py
```

Run `scripts/get_pretrained.py` script to download pre-trained model in the `saved/models/pretrained` directory. (Model was trained with `src/configs/train.json`)

## Project structure

Repository is structured in the following way.

-   `src` includes code for all used objects and functions including:

    -   `base`: base class code for models and trainers in the correspongding `base_name.py` file.

    -   `collate_fn`: code for the corresponding function of the dataloader.

    -   `configs`: configs for model training in `config_name.json`.

    -   `datasets`: code for LJSpeech dataset.

    -   `logger`: code for different loggers (including W&B) and some utils for visualization.

    -   `loss`: definition of loss function for HiFiGAN with optimizers step inside.

    -   `model`: code for HiFiGAN model and its subparts in `blocks` subdir.

    -   `preprocessing`: code for mel spectrogram extraction.

    -   `trainer`: code for training models.

    -   `utils`: basic utils including `parse_config.py` for parsing `.json` configs, `object_loading.py` for dataloaders structuring and `util.py` for files\device control and metrics wrapper.

-   `data` folder consists of downloaded datasets folders created by running `download_data.py` script.

-   `saved` folder consists of logs and model checkpoints \ their configs in `log` and `models` subdirs respectively.

-   `scripts` folder consists of different `.py` files for downloading data, feature extracting, results logging and pre-trained models.

-   `results` folder consists of synthesized test speech.

-   `requirements.txt` includes all packages required to run the project.

-   `train.py` script for training models.

-   `synthesis.py` script for synthesizing speech.

## Training

To train model run the following script:

```
python3 train.py -c src/configs/config_name.json
```

To resume training from checkpoint (resume optimizers, schedulers etc.)

```
python3 train.py -r path\to\saved\checkpoint.pth
```

To train model with initialization from checkpoint:

```
python3 train.py -c src/configs/config_name.json \
    -p path\to\saved\checkpoint.pth
```

## Synthesizing

To synthesize test utterances run the following command:

```bash
python3 synthesis.py -c path/to/saved/config -p path/to/model/checkpoint
```

For pre-trained model it will be:

```bash
python3 synthesis.py -c saved/models/pretrained/final/config.json\
        -p saved/models/pretrained/final/model_best.pth
```

The sythesizer saves audio into `results` directory in the following format:

```
t={text_index_in_test_list}.wav
```

Default test phrases:

-   A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest

-   Massachusetts Institute of Technology may be best known for its math, science and engineering education

-   Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space

To log results directory to WandB run `scripts/log_results_wandb.py`. See `--help` option for setting project and run names.

## Authors

-   Petr Grinberg

## License

This project is licensed under the [MIT](LICENSE) License - see the [LICENSE](LICENSE) file for details.

## Credits

Project template was taken from [ASR repository](https://github.com/Blinorot/ASR). MelSpectrogram function was taken from [DLA repository](https://github.com/markovka17/dla/tree/2022/hw4_nv)
