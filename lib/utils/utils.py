# encoding: utf-8
import os
import json
import pickle
import logging
import sys
from typing import Any, Dict, Optional

def mkdir_if_missing(dir_path: str) -> None:
    os.makedirs(dir_path, exist_ok=True)


def read_json(fpath: str) -> Dict:
    with open(fpath, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(obj: Dict, fpath: str) -> None:
    mkdir_if_missing(os.path.dirname(fpath))
    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def save_pickle(obj: Any, fpath: str) -> None:
    mkdir_if_missing(os.path.dirname(fpath))
    with open(fpath, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(fpath: str) -> Any:
    with open(fpath, 'rb') as f:
        return pickle.load(f)



def setup_logger(name: str, save_dir: Optional[str] = None, is_train: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        return logger

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        filename = "train_log.txt" if is_train else "test_log.txt"
        fh = logging.FileHandler(os.path.join(save_dir, filename), mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
