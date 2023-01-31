import numpy as np
import pickle
from typing import Tuple


def load_pickle(
    path: str, flatten: bool = False, binarize: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    train_data = data["train"].astype("float32")
    test_data = data["test"].astype("float32")
    if binarize:
        train_data = (train_data > 128).astype("float32")
        test_data = (test_data > 128).astype("float32")
    else:
        train_data = train_data / 255.0
        test_data = test_data / 255.0
    train_data = np.transpose(train_data, (0, 3, 1, 2))
    test_data = np.transpose(test_data, (0, 3, 1, 2))
    if flatten:
        train_data = train_data.reshape(len(train_data.shape[0]), -1)
        test_data = test_data.reshape(len(train_data.shape[0]), -1)
    return train_data, test_data
