from pathlib import Path

import requests
from tqdm import tqdm

from oml.const import PROJECT_ROOT
from oml.utils.io import check_exists_and_validate_md5

DALI_DATASET_HASH = '81b083711301e6f88072f4c288978b3d'
DALI_DATASET_DIR = PROJECT_ROOT / "data/images"


def download_dali_dataset(save_dir: Path = DALI_DATASET_DIR):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset_url = "https://raw.githubusercontent.com/NVIDIA/DALI/main/docs/examples/data/images/"
    file_list_fname = "file_list.txt"
    files_list = requests.get(dataset_url + file_list_fname)
    assert files_list.status_code == 200, files_list.status_code
    files_list = files_list.content.decode()

    with open(save_dir / file_list_fname, "w+") as fout:
        fout.write(files_list)

    files_list = [Path(line.split(" ")[0]) for line in files_list.splitlines()]

    for file in tqdm(files_list):
        (save_dir / file.parent).mkdir(parents=True, exist_ok=True)
        image = requests.get(dataset_url + str(file))
        assert image.status_code == 200, image.status_code
        with open(save_dir / file, "wb+") as fout:
            fout.write(image.content)

    print(f"DALI images dataset saved to {save_dir}")


def download_dali_if_necessary(save_dir: Path = DALI_DATASET_DIR):
    if not check_exists_and_validate_md5(save_dir, DALI_DATASET_HASH):
        download_dali_dataset(save_dir)
    else:
        print('DALI dataset already on disk')


