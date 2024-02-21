import functools
import os

import dill
import numpy as np

from fileutil import dircheck, try_get_cached_vals, cache_vals
from imgutil import compute_vals


def curry_size(f):
    @functools.wraps(f)
    def wrapper(domain_size):
        return functools.partial(f, domain_size=domain_size)

    return wrapper


def demo_trig(pos, domain_size) -> np.float64:
    """Combination of sin and cos."""
    return np.sin(pos[0] / 10) * np.cos(pos[1] / 10) - 0.5 * np.sin(pos[0] / 20) + 0.5 * np.cos(pos[1] / 20)


def demo_waves(pos, domain_size) -> np.float64:
    """Looks a bit like waves."""
    return (
        np.sin(pos[0] / 10) * np.cos(pos[1] / 10)
        + 1.2 * np.sin(pos[0] / 5)
        + (pos[1] - domain_size / 2) ** 2 / 1e5
        + (pos[0] - domain_size / 2) ** 2 / 1e5
    )


def demo_hole(pos, domain_size) -> np.float64:
    """Just a simple quadratic function."""
    return (pos[0] / 10 - domain_size / 20) ** 2 + (pos[1] / 10 - domain_size / 20) ** 2


def demo_infs(pos, domain_size) -> np.float64:
    """A function with infinities to show how the background gradient renderer handles them."""
    x = np.sum(pos)
    if x < domain_size / 2:
        return np.float64(np.inf)
    elif x > domain_size * 1.5:
        return np.float64(-np.inf)
    else:
        return np.sin(x / 10) + np.cos(x / 20)


def save_demos(precompute_size=None) -> None:
    """Pickle the demo functions, precompute their values if `precompute_size` is not None."""
    functions = [demo_trig, demo_waves, demo_hole, demo_infs]
    dircheck("demos")
    for f in functions:
        with open(os.path.join("demos", f"{f.__name__}.pkl"), "wb") as file:
            dill.dump(curry_size(f), file)
    if precompute_size is not None:
        for f in functions:
            if try_get_cached_vals(os.path.join("demos", f"{f.__name__}.pkl"), precompute_size) is None:
                print(f"Precomputing values for {f.__name__}...")
                vals = compute_vals(curry_size(f)(precompute_size), precompute_size, precompute_size)
                cache_vals(os.path.join("demos", f"{f.__name__}.pkl"), vals)


if __name__ == "__main__":
    save_demos()
