import os
import time
import urllib.error
import urllib.request

import numpy as np
import PIL.Image

from fileutil import dircheck, try_get_cached_vals, cache_vals
from imgutil import BitmapFun, compute_vals


def save_demos(precompute_size=None) -> None:
    """Download demo images, precompute values of their `BitmapFun`s if `precompute_size` is not None."""
    web_images = [
        # random streaks I made in GIMP
        ("https://i.imgur.com/SDaWwRC.png", "random_paint.png"),
        # perlin noise made in http://kitfox.com/projects/perlinNoiseMaker/
        ("https://i.imgur.com/EqjUPLB.png", "perlin_noise.png"),
        # tiny noise texture I made in GIMP
        ("https://i.imgur.com/B0aj7yx.png", "noise_small.png"),
        # public domain (to my knowledge)
        ("https://i.imgur.com/t7tgaRG.jpeg", "kitten_drawing.jpg")
    ]
    dircheck("demos")
    print("Downloading demo images...")
    for url, filename in web_images:
        if not os.path.isfile(os.path.join("demos", filename)):
            print(f"Downloading {filename} from {url}...")
            try:
                urllib.request.urlretrieve(url, os.path.join("demos", filename))
            except urllib.error.HTTPError as e:
                print(f"Failed to download {filename}: {e}")
    if precompute_size is not None:
        print(f"Precomputing {precompute_size}x{precompute_size} values for...")
        start_t = time.time()
        for _, img_file in web_images:
            img_path = os.path.join("demos", img_file)
            if not os.path.isfile(img_path):
                continue
            if try_get_cached_vals(img_path, precompute_size) is None:
                print(f"{img_file}...")
                with PIL.Image.open(img_path) as img:
                    vals = compute_vals(
                        BitmapFun(np.asarray(img.convert("L")), precompute_size), precompute_size, precompute_size
                    )
                cache_vals(img_path, vals)
        print(f"Precomputing took {time.time() - start_t:.2f} seconds.")


if __name__ == "__main__":
    save_demos()
