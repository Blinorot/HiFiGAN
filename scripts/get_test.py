import os
import shutil
from pathlib import Path

import gdown

URL_LINKS = {
    "test_data": "https://drive.google.com/u/0/uc?id=1N5qIvWQNMnmYzEfErHNJjSHjR_oY-1Qb&export=download",
}

def download():
    data_dir = Path(__file__).absolute().resolve().parent.parent
    data_dir = data_dir / "data"
    data_dir.mkdir(exist_ok=True, parents=True)

    arc_path = data_dir / 'test_data.zip'
    if not arc_path.exists():
        gdown.download(URL_LINKS['test_data'], str(arc_path))
    shutil.unpack_archive(str(arc_path), str(data_dir))
    os.remove(str(arc_path))

if __name__ == '__main__':
    download()
