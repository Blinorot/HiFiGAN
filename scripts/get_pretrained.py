import os
import shutil
from pathlib import Path

import gdown

URL_LINKS = {
    "train": "https://drive.google.com/u/0/uc?id=1uFViHhwT7ElmJSCyMy5mWZvznS8kNuY-&export=download",
    "DS_train": "https://drive.google.com/u/0/uc?id=1LtYzLYUjTsZIImb5chlZSJy_F8bu-ebq&export=download",
}

def download():
    data_dir = Path(__file__).absolute().resolve().parent.parent
    data_dir = data_dir / "saved" / "models" / "pretrained"
    data_dir.mkdir(exist_ok=True, parents=True)

    arc_path = data_dir / 'train.zip'
    if not arc_path.exists():
        gdown.download(URL_LINKS['train'], str(arc_path))
    shutil.unpack_archive(str(arc_path), str(data_dir))
    os.remove(str(arc_path))

    arc_path = data_dir / 'DS_train.zip'
    if not arc_path.exists():
        gdown.download(URL_LINKS['DS_train'], str(arc_path))
    shutil.unpack_archive(str(arc_path), str(data_dir))
    os.remove(str(arc_path))

if __name__ == '__main__':
    download()
