from pathlib import Path

import requests
from tqdm import tqdm

from oml.const import PROJECT_ROOT


def get_data(save_dir: Path = PROJECT_ROOT / "data/images"):
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


get_data()
