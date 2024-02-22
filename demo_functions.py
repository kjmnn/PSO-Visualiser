import functools
import os
import time

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
    """A function with infinities that showcases how the background gradient renderer handles them."""
    x = np.sum(pos)
    if x < domain_size / 2:
        return np.float64(np.inf)
    elif x > domain_size * 1.5:
        return np.float64(-np.inf)
    else:
        return np.sin(x / 10) + np.cos(x / 20)

# Now for some actual real benchmark functions:
# Adapted from https://en.wikipedia.org/wiki/Test_functions_for_optimization
# on 2024/02/22

def demo_rastrigin(pos, domain_size) -> np.float64:
    """Rastrigin function."""
    y, x = (pos / domain_size) * 10.24 - 5.12
    return 20 + x**2 + y**2 - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))


def demo_ackley(pos, domain_size) -> np.float64:
    """Ackley function."""
    y, x = (pos / domain_size) * 10 - 5
    return (
        -20 * np.exp(-0.5 * np.sqrt(0.5 * (x**2 + y**2)))
        - np.exp(0.5 * (np.cos(np.pi * 2 * x) + np.cos(np.pi * 2 * y)))
        + np.e
        + 20
    )

# This one looks a bit off, but it still works fine
def demo_eggholder(pos, domain_size) -> np.float64:
    """Eggholder function."""
    y, x =  1024 * pos / domain_size - 512
    return -(y + 47) * np.sin(np.sqrt(np.abs(x / 2 + y + 47))) - x * np.sin(np.sqrt(np.abs(x - (y + 47))))


def demo_himmelblau(pos, domain_size) -> np.float64:
    """Himmelblau's function."""
    y, x = (pos / domain_size) * 10 - 5
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


def save_demos(precompute_size=None) -> None:
    """Pickle the demo functions, precompute their values if `precompute_size` is not None."""
    functions = [demo_trig, demo_waves, demo_hole, demo_infs, demo_rastrigin, demo_ackley, demo_eggholder, demo_himmelblau]
    dircheck("demos")
    print("Pickling demo functions...")
    for f in functions:
        with open(os.path.join("demos", f"{f.__name__}.pkl"), "wb") as file:
            dill.dump(curry_size(f), file)
    if precompute_size is not None:
        print(f"Precomputing {precompute_size}x{precompute_size} values for...")
        start_t = time.time()
        for f in functions:
            if try_get_cached_vals(os.path.join("demos", f"{f.__name__}.pkl"), precompute_size) is None:
                print(f"{f.__name__}.pkl...")
                vals = compute_vals(curry_size(f)(precompute_size), precompute_size, precompute_size)
                cache_vals(os.path.join("demos", f"{f.__name__}.pkl"), vals)
        print(f"Precomputing took {time.time() - start_t:.2f} seconds.")


if __name__ == "__main__":
    save_demos()
