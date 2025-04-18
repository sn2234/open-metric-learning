import inspect
import os
import random
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Sequence,
    Tuple,
    Union,
)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from oml.const import BLACK, TCfg, TColor


class CompatibilityError(Exception):
    pass


def adapt_argument_as_kwarg(obj: Callable, candidates: List[str], value: Any) -> Dict[str, Any]:  # type: ignore
    parameters = list(inspect.signature(obj).parameters.keys())

    for name in candidates:
        if name in parameters:
            return {name: value}

    raise CompatibilityError(
        f"Not compatible signature for {obj}. Candidates: {candidates}, available arguments: {parameters}"
    )


def find_value_ids(it: Iterable[Any], value: Any) -> List[int]:
    """
    Args:
        it: List of any
        value: Query element

    Returns:
        Indices of the all elements equal to x0
    """
    if isinstance(it, np.ndarray):
        inds = list(np.where(it == value)[0])
    else:  # could be very slow
        inds = [i for i, el in enumerate(it) if el == value]
    return inds


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    os.environ["PL_SEED_WORKERS"] = str(1)

    try:
        import torch_xla.core.xla_model as xm
    except ImportError:
        pass
    else:
        xm.set_rng_state(seed)


def one_hot(i: int, dim: int) -> torch.Tensor:
    vector = torch.zeros(dim)
    vector[i] = 1
    return vector


def flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "/", ignored_keys: Iterable[str] = ()
) -> Dict[str, Any]:
    items = []  # type: ignore
    for k, v in d.items():
        if k in ignored_keys:
            continue
        new_key = str(parent_key) + sep + str(k) if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep, ignored_keys=ignored_keys).items())
        else:
            items.append((new_key, v))
    return dict(items)


def dictconfig_to_dict(cfg: TCfg) -> Dict[str, Any]:
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)

    ret = dict()

    for k in cfg.keys():

        if isinstance(cfg[k], DictConfig) or isinstance(cfg[k], dict):
            ret[k] = dictconfig_to_dict(cfg[k])
        else:
            ret[k] = cfg[k]

    return ret


def smart_sample(array: List[Any], k: int) -> List[Any]:
    """Sample n_samples items from given list. If array contains at least n_samples items, sample without repetition;
    otherwise take all the unique items and sample n_samples - len(array) ones with repetition.

    Args:
        array: list of unique elements to sample from
        k: number of items to sample

    Returns:
        sampled_items: list of sampled items
    """
    array_size = len(array)
    if array_size < k:
        sampled = (
            np.random.choice(array, size=array_size, replace=False).tolist()
            + np.random.choice(array, size=k - array_size, replace=True).tolist()
        )
    else:
        sampled = np.random.choice(array, size=k, replace=False).tolist()
    return sampled


def clip_max(arr: Tuple[int, ...], max_el: int) -> Tuple[int, ...]:
    return tuple(min(x, max_el) for x in arr)


def remove_unused_kwargs(kwargs: Dict[str, Any], function: Any) -> Dict[str, Any]:
    return {k: v for k, v in kwargs.items() if k in inspect.signature(function).parameters}


def check_if_nonempty_positive_integers(var: Union[int, Sequence[int]], name: str) -> None:
    """
    Check whether ``var`` is a positive integer or a non-empty Iterable of positive integers.

    Args:
        var: A sequence.
        name: A name of the sequence in case of exception should be raised.

    """
    if isinstance(var, Sequence):
        if not len(var) > 0 or not all([isinstance(x, int) and (x > 0) for x in var]):
            raise ValueError(f"{name} is expected to be non-empty and contain positive integers, but got {var}")
    elif isinstance(var, int):
        if var <= 0:
            raise ValueError(f"{name} is expected to be a positive integer, but got {var}")
    else:
        raise ValueError(f"Unsupported argument type. Expected int or Iterable[int], but got {type(var)}")


def compare_dicts_recursively(d1: Dict, d2: Dict) -> bool:  # type: ignore
    """
    The function compares dictionaries and prints the exact information where they differ. By using the built-in
    dictionary comparison one can get just a plain 'True' or 'False' as result of the comparison, without any hints
    on where the dictionaries differ.
    """
    assert set(d1.keys()) == set(
        d2.keys()
    ), f"The dictionaries keys are different.\nDict_1 keys: {set(d1.keys())}\nDict_2 keys: {set(d2.keys())}"
    for k, v in d1.items():
        if isinstance(v, dict):
            assert compare_dicts_recursively(
                v, d2[k]
            ), f"The dictionaries differs at key {k}.\nDict_1 value: {v}\nDict_2 value: {d2[k]}"
        elif isinstance(v, Tensor):
            assert torch.allclose(d2[k], v)
        else:
            assert d2[k] == v, f"Key name: {k}\nDict_1 value: {v}\nDict_2 value: {d2[k]}"
    return True


def pad_array_right(arr: np.ndarray, required_len: int, val: Union[float, int]) -> np.ndarray:
    assert required_len >= len(arr)
    assert arr.ndim == 1

    return np.pad(arr, (0, required_len - len(arr)), mode="constant", constant_values=val)


def visualize_text(text: str, color: TColor = BLACK, draw_bbox: bool = True) -> np.ndarray:
    with matplotlib_backend("Agg"):
        fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)
        ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True, fontsize=20)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image = image.reshape(height, width, 4)[..., :3]
        plt.close(fig)

    image = np.resize(image, (256, 256, 3))

    if draw_bbox:
        frame_thickness = 5
        image[:frame_thickness, :, :] = color  # Top frame
        image[-frame_thickness:, :, :] = color  # Bottom frame
        image[:, :frame_thickness, :] = color  # Left frame
        image[:, -frame_thickness:, :] = color  # Right frame

    return image


def html_blocks_to_grid(blocks: List[List[str]]) -> str:
    html_rows = []
    for row in blocks:
        row_html = "<div style='display:flex; justify-content:left;'>" + "".join(row) + "</div>"
        html_rows += [row_html]

    final_html = "<div>" + "\n".join(html_rows) + "</div>"

    return final_html


@contextmanager
def matplotlib_backend(backend: str) -> Generator[None, None, None]:
    current_backend = matplotlib.get_backend()
    try:
        matplotlib.use(backend)
        yield
    finally:
        matplotlib.use(current_backend)


__all__ = [
    "CompatibilityError",
    "adapt_argument_as_kwarg",
    "find_value_ids",
    "set_global_seed",
    "one_hot",
    "flatten_dict",
    "dictconfig_to_dict",
    "smart_sample",
    "clip_max",
    "check_if_nonempty_positive_integers",
    "compare_dicts_recursively",
    "pad_array_right",
    "visualize_text",
    "matplotlib_backend",
    "html_blocks_to_grid",
]
