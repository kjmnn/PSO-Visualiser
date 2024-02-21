import hashlib
import lzma
import os
from typing import Optional

import dill
import numpy as np
import numpy.typing as npt


def dircheck(*ds) -> None:
    """Ensure that a directory exists."""
    d = os.path.join(*ds)
    if not os.path.exists(d):
        os.makedirs(d)


def hashcmp(file: str, hash_file: str) -> bool:
    """Compare `file`'s sha256 hash with hash stored in `hash_file`."""
    with open(file, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    with open(hash_file, "r") as f:
        return file_hash == f.read()


def save_hash(file: str, hash_file: str) -> None:
    """Save `file`'s sha256 hash to `hash_file`."""
    with open(file, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    with open(hash_file, "w") as f:
        f.write(file_hash)


def try_get_cached_vals(in_path: str, size: int) -> Optional[npt.NDArray[np.float64]]:
    """
    Try to load cached values, return None if not found.
    Cached values are looked up based on the input file name and the precomputed array's size,
    meaning that different files with the same name will conflict.
    The file's hash is used to check input file identity.
    """
    _, in_file_name = os.path.split(in_path)
    in_hash = os.path.join(".cache", "hashes", in_file_name + ".sha256")
    cached_vals = os.path.join(".cache", "vals", in_file_name + f"_{size}x{size}.pkl")
    something_in_cache = os.path.isfile(in_hash) and os.path.isfile(cached_vals)
    if something_in_cache and hashcmp(in_path, in_hash):
        with lzma.open(cached_vals, "rb") as file:
            return dill.load(file)
    else:
        return None


def cache_vals(in_path: str, vals: npt.NDArray[np.float64]) -> None:
    """
    Save `vals` to a file in the cache, along with a hash of the file at `in_path`.
    Since a of the input file's contents is used to check identity, 
    caching values for a different size will not invalidate values precomputed for different sizes.
    """
    _, in_file_name = os.path.split(in_path)
    cache_hashes = os.path.join(".cache", "hashes")
    cache_vals = os.path.join(".cache", "vals")
    dircheck(cache_hashes)
    dircheck(cache_vals)
    m, n = vals.shape
    save_hash(in_path, os.path.join(cache_hashes, in_file_name + ".sha256"))
    with lzma.open(os.path.join(cache_vals, in_file_name + f"_{m}x{n}.pkl"), "wb") as file:
        dill.dump(vals, file)


if __name__ == "__main__":
    print("This is just a library, running it does nothing.")
