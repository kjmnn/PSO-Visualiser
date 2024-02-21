from dataclasses import dataclass
from typing import Callable, Literal, List, Optional

import numpy as np
import numpy.typing as npt


@dataclass(slots=True)
class Particle:
    """Dataclass storing data of a single particle."""

    pos: npt.NDArray[np.float64]
    v: npt.NDArray[np.float64]
    best_val: np.float64
    best_pos: npt.NDArray[np.float64]

    def __init__(
        self,
        pos: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
        f: Callable[[npt.NDArray[np.float64]], np.float64],
    ):
        self.pos = np.copy(pos)
        self.v = np.copy(v)
        self.best_val = f(pos)
        self.best_pos = np.copy(pos)


class Simulation:
    """
    The PSO algorithm simulates a swarm of particles moving through the optimisation search space.
    Currently restricted to 2D spaces.
    """

    best_val: np.float64
    best_pos: npt.NDArray[np.float64]
    particles: List[Particle]
    _f_raw: Callable[[npt.NDArray], np.float64]
    _maximise: bool
    _init_v: np.float64
    _inertia: np.float64
    # learning rates
    _lr_p: np.float64
    _lr_g: np.float64
    _random: np.random.Generator
    _sim_size: int

    def __init__(
        self,
        seed: int,
        f: Callable[[npt.NDArray[np.float64]], np.float64],
        pop_shape: Literal["random", "grid"],
        pop: int,
        maximise: bool,
        inertia: np.float64,
        lr_p: np.float64,
        lr_g: np.float64,
        init_v: np.float64,
        sim_size: int,
    ):
        self._random = np.random.default_rng(seed)
        self._sim_size = sim_size
        self._f_raw = f
        self._maximise = maximise
        self._init_v = init_v
        self._inertia = inertia
        self._lr_p = lr_p
        self._lr_g = lr_g
        if pop_shape == "random":
            particle_positions = self._random.uniform(0, sim_size, (pop, 2))
        elif pop_shape == "grid":
            k = int(np.ceil(np.sqrt(pop)))
            particle_positions = [
                np.array([x, y])
                for x in np.linspace(0, sim_size, k, endpoint=True)
                for y in np.linspace(0, sim_size, k, endpoint=True)
            ]
        else:
            raise ValueError("Invalid pop_shape. Must be 'random' or 'grid'.")
        self.particles = [
            Particle(pos, self._random.uniform(-init_v, init_v, 2), self._f) for pos in particle_positions
        ]
        vals = [f(part.pos) for part in self.particles]
        best_i = np.argmin(vals)
        self.best_val = vals[best_i]
        self.best_pos = self.particles[best_i].pos.copy()

    def step(self) -> None:
        """Calculate a step of the PSO algorithm."""
        for part in self.particles:
            part.v *= self._inertia
            part.v += (part.best_pos - part.pos) * self._random.uniform(0, 1) * self._lr_p
            part.v += (self.best_pos - part.pos) * self._random.uniform(0, 1) * self._lr_g
            part.pos += part.v
            val = self._f(part.pos)
            if val < part.best_val:
                # new local best
                part.best_val = val
                part.best_pos = np.copy(part.pos)
            if val < self.best_val:
                # new global best
                self.best_val = val
                self.best_pos = np.copy(part.pos)

    def best(self) -> tuple[np.float64, npt.NDArray[np.float64]]:
        """Return the best value and position found."""
        best_val = self.best_val if not self._maximise else -self.best_val
        return best_val, self.best_pos

    def spawn_particle(self, pos: Optional[npt.NDArray[np.float64]] = None) -> None:
        """Spawn a new particle at `pos` (or randomly if `pos` is `None`)."""
        if pos is None:
            pos = self._random.uniform(0, self._sim_size, 2)
        else:
            pos = pos.astype(np.float64)
        part = Particle(pos, self._random.uniform(-self._init_v, self._init_v, 2), self._f)
        self.particles.append(part)

    def _f(self, pos: npt.NDArray[np.float64]) -> np.float64:
        """
        Evaluate the function being optimised at `pos`.
        If `self._maximise` is True, the value will be negated (to avoid negating elsewhere).
        If `pos` is out of bounds, returns infinity.
        """
        if (pos < 0).any() or (pos > self._sim_size).any():
            return np.float64("inf")
        val = self._f_raw(pos)
        return val if not self._maximise else -val


if __name__ == "__main__":
    print("This is just a library, running it does nothing.")
