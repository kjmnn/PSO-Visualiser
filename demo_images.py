import os
import urllib.request

import numpy as np
import PIL.Image

from fileutil import dircheck, try_get_cached_vals, cache_vals
from imgutil import BitmapFun, compute_vals


def save_demos(precompute_size=None) -> None:
    """Download demo images, precompute values of their `BitmapFun`s if `precompute_size` is not None."""
    web_images = [
        ("https://www.python.org/static/community_logos/python-logo-master-v3-TM.png", "python.png"),
    ]
    dircheck("demos")
    for url, filename in web_images:
        if not os.path.isfile(os.path.join("demos", filename)):
            print(f"Downloading {filename} from {url}...")
            urllib.request.urlretrieve(url, os.path.join("demos", filename))
    if precompute_size is not None:
        for _, img_file in web_images:
            img_path = os.path.join("demos", img_file)
            if not os.path.isfile(img_path):
                continue
            if try_get_cached_vals(img_path, precompute_size) is None:
                print(f"Precomputing values for {img_file}...")
                with PIL.Image.open(img_path) as img:
                    vals = compute_vals(
                        BitmapFun(np.asarray(img.convert("L")), precompute_size), precompute_size, precompute_size
                    )
                cache_vals(img_path, vals)


if __name__ == "__main__":
    save_demos()
