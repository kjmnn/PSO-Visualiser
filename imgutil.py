from typing import Callable, Optional

import numpy as np
import numpy.typing as npt
import PIL.Image


class BitmapFun:
    """
    Scalar function defined by a bitmap.
    Pixels are converted to greyscale and interpolated
    (so the image is in effect squished / stretched to fit the domain).
    """

    _bitmap: npt.NDArray[np.uint16]
    _domain_size: int
    _scale: npt.NDArray[np.float64]

    def __init__(self, bitmap: npt.NDArray, domain_size: int):
        # extend the bitmap to avoid glitches around the domain border
        self._bitmap = np.pad(bitmap, ((0, 2), (0, 2)), mode="edge")
        self._domain_size = domain_size
        self._scale = np.array(bitmap.shape) / domain_size

    def __call__(self, pos: npt.NDArray[np.float64]) -> np.float64:
        if (pos < 0).any() or (pos > self._domain_size).any():
            return np.float64("nan")
        # rescale `pos` from domain to bitmap coordinates
        y, x = pos * self._scale
        # split coordinates into integer and decimal parts
        yi, xi = np.floor([y, x]).astype(int)
        yd, xd = np.mod([y, x], 1)
        # interpolate
        return np.dot(
            self._bitmap[yi : yi + 2, xi : xi + 2].flatten(),
            [(1 - yd) * (1 - xd), (1 - yd) * xd, yd * (1 - xd), yd * xd],
        )


def finitise(vals: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Replace infinities by numbers larger / smaller than the maximum / minimum finite value.
    """
    finites = vals[np.isfinite(vals)]
    min_val, max_val = np.min(finites), np.max(finites)
    if min_val == max_val:
        diff = 1
        vals = np.where(np.isinf(vals), vals, 0)
    else:
        diff = max_val - min_val
    vals[np.isposinf(vals)] = max_val + diff / 2
    vals[np.isneginf(vals)] = min_val - diff / 2
    return vals


def compute_vals(
    f: Callable[[npt.NDArray[np.float64]], np.float64], width: int, height: int
) -> npt.NDArray[np.float64]:
    """
    Evaluate a function at each pixel.
    It's pretty slow, but I don't think there's a a way to make it faster.
    """
    return np.array([[f(np.array([y, x])) for x in range(width)] for y in range(height)])


def vals_to_image(
    vals: npt.NDArray[np.float64],
    rgb_low: npt.NDArray[np.uint8],
    rgb_high: npt.NDArray[np.uint8],
):
    """
    Create a gradient image from precalculated function values.
    """
    nans = np.isnan(vals)
    if nans.any():
        print(f"Warning: NaNs at {np.sum(nans)} positions, replacing with 0.")
        vals[nans] = 0
    if np.isinf(vals).any():
        vals = finitise(vals)

    min_val, max_val = vals.min(), vals.max()
    if min_val == max_val:
        vals = np.zeros_like(vals)
    else:
        vals = (vals - min_val) / (max_val - min_val)

    oklab_blend = True
    if oklab_blend:
        Ll, al, bl = rgb_to_oklab(rgb_low)
        Lh, ah, bh = rgb_to_oklab(rgb_high)
        L = Ll + vals * (Lh - Ll)
        a = al + vals * (ah - al)
        b = bl + vals * (bh - bl)
        rgb = oklab_to_rgb(L, a, b)
    else:
        # rgb blend for comparison
        rgb_low_ = rgb_low / 255
        rgb_high_ = rgb_high / 255
        rgb = (rgb_low_ + vals[..., np.newaxis] * (rgb_high_ - rgb_low_))

    return PIL.Image.fromarray((rgb * 255).round().astype(np.uint8), "RGB")


def try_parse_rgb(rgb_hex: str) -> Optional[npt.NDArray[np.uint8]]:
    """Parse a RGB hex string."""
    if len(rgb_hex) < 6:
        return None
    if rgb_hex[0] == "#":
        rgb_hex = rgb_hex[1:]
    if rgb_hex[:2].lower() == "0x":
        rgb_hex = rgb_hex[2:]
    if len(rgb_hex) != 6:
        return None
    try:
        return np.array([int(rgb_hex[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)
    except ValueError:
        return None


def rgb_to_tk_hex(rgb: npt.NDArray[np.uint8]) -> str:
    """Convert RGB array to a tk-compatible hex string."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def rgb_to_oklab(rgb: npt.NDArray[np.uint8]) -> tuple[np.float64, np.float64, np.float64]:
    """
    Convert from sRGB to Oklab.
    See https://bottosson.github.io/posts/oklab/ for explanation and original code.
    The parameter is a vector of 8-bit sRGB values.
    """
    # 8bit sRGB to linear
    rgb_ = rgb / 255
    r, g, b = np.where(rgb_ >= 0.04045, ((rgb_ + 0.055) / 1.055) ** 2.4, rgb_ / 12.92)

    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    l_ = l ** (1 / 3)
    m_ = m ** (1 / 3)
    s_ = s ** (1 / 3)

    return (
        np.float64(0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_),
        np.float64(1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_),
        np.float64(0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_),
    )


def oklab_to_rgb(
    L: npt.NDArray[np.float64], a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Convert from Oklab to sRGB.
    See https://bottosson.github.io/posts/oklab/ for explanation and original code.
    The parameters are separate arrays of L, a, and b values because it's more efficient to compute them all at once (I hope).
    """

    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    l = l_**3
    m = m_**3
    s = s_**3

    rgb = np.zeros(L.shape + (3,), dtype=np.float64)
    rgb[:, :, 0] = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    rgb[:, :, 1] = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    rgb[:, :, 2] = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    np.clip(rgb, 0, np.inf, out=rgb)
    # linear to floating sRGB
    rgb = np.where(rgb >= 0.0031308, 1.055 * rgb ** (1 / 2.4) - 0.055, 12.92 * rgb)
    np.clip(rgb, 0, 1, out=rgb)
    return rgb


if __name__ == "__main__":
    print("This is just a library, running it does nothing.")
