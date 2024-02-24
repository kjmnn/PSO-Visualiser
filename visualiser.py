import argparse
import functools  # for pickled functions
import os
import time
import tkinter as tk
from typing import Callable, Literal, List, Optional, Tuple

import dill
import numpy as np
import numpy.typing as npt
import PIL.Image, PIL.ImageTk

from pso import Simulation
from fileutil import try_get_cached_vals, cache_vals
from imgutil import compute_vals, BitmapFun, vals_to_image, try_parse_rgb, rgb_to_tk_hex


class App:
    """Interactive app that runs the `PSO` algorithm providing a tkinter GUI."""

    ## Simulation parameters
    _f: Callable[[npt.NDArray], np.float64]
    _maximise: bool
    _inertia: np.float64
    _p_coef: np.float64
    _g_coef: np.float64
    _pop_shape: Literal["random", "grid"]
    _pop: int
    _seed: int
    _init_v: np.float64
    _sim_size: int
    ## App parameters
    _tps: np.float64
    # increment seed before each restart
    _increment_seed: bool
    # amount of particles to spawn on click
    _spawn_amount: int
    # the simulation itself
    _sim: Simulation
    # tk id of the next tick task
    _next_tick: Optional[str]
    ## GUI
    _window: tk.Tk
    _canvas: tk.Canvas
    _best_label: tk.Label
    # tk vars for input fields, parsers & names of their respective app vars
    _tkvar_bindings: List[Tuple[tk.StringVar, Callable, str]]
    # bound for easy incrementing
    _seed_tkvar: tk.StringVar
    # identifers of drawn particle circles
    _drawn_particles: List[int]
    _particle_colour: str
    _particle_radius: int
    # base line thickness for various borders
    _line_thickness: int
    # precalculated offset of particle circle centres
    _particle_offset: int
    # background image (function visualisation)
    _bg: PIL.Image.Image
    # the actual tk object (stored to prevent garbage collection)
    _bg_obj: PIL.ImageTk.PhotoImage

    def __init__(
        self,
        seed: int,
        f: Callable[[npt.NDArray[np.float64]], np.float64],
        maximise: bool,
        pop_type: Literal["random", "grid"],
        pop: int,
        inertia: np.float64,
        p_coef: np.float64,
        g_coef: np.float64,
        init_v: np.float64,
        sim_size: int,
        tps: np.float64,
        particle_colour: str,
        particle_radius: int,
        line_thickness: int,
        bg: PIL.Image.Image,
        increment_seed: bool = False,
        spawn_amount: int = 1,
    ):
        self._seed = seed
        self._f = f
        self._maximise = maximise
        self._pop_shape = pop_type
        self._pop = pop
        self._inertia = inertia
        self._p_coef = p_coef
        self._g_coef = g_coef
        self._init_v = init_v
        self._sim_size = sim_size
        self._tps = tps
        colour_maybe = try_parse_rgb(particle_colour)
        if colour_maybe is not None:
            self._particle_colour = rgb_to_tk_hex(colour_maybe)
        else:
            self._particle_colour = "#000000"
        self._particle_radius = particle_radius
        self._line_thickness = line_thickness
        self._bg = bg
        self._particle_offset = particle_radius + line_thickness // 2
        self._increment_seed = increment_seed
        self._spawn_amount = spawn_amount
        self._next_tick = None

    def _init_sim(self) -> None:
        self._sim = Simulation(
            self._seed,
            self._f,
            self._pop_shape,
            self._pop,
            self._maximise,
            self._inertia,
            self._p_coef,
            self._g_coef,
            self._init_v,
            self._sim_size,
        )

    def _init_gui(self) -> None:
        """
        Initialise the GUI.
        Creates a window with a canvas for visualisation and a column of input fields and buttons.
        """
        border_width = np.ceil(self._line_thickness * 1.5)
        self._window = tk.Tk()
        self._window.title("PSO Visualiser")
        self._window.resizable(False, False)
        self._window.protocol("WM_DELETE_WINDOW", self._quit)
        self._window.bind("<Control-q>", self._quit)
        self._window.bind("<Control-r>", self._restart_sim)

        canvas_frame = tk.Frame(self._window, bd=border_width, relief="solid")
        canvas_frame.grid(row=0, column=0)

        self._canvas = tk.Canvas(canvas_frame, width=self._sim_size, height=self._sim_size)
        self._canvas.grid(row=0, column=0)
        # bind to prevent garbage collection
        self._bg_obj = PIL.ImageTk.PhotoImage(self._bg)
        self._canvas.create_image(1, 1, image=self._bg_obj, anchor=tk.NW)
        self._drawn_particles = []
        self._canvas.bind("<Button-1>", self._spawn_particles_on_click)

        controls_frame = tk.Frame(self._window, bd=border_width, relief="flat")
        controls_frame.grid(row=0, column=1, sticky=tk.NS)

        self._tkvar_bindings = []

        def with_min(convert, min_) -> Callable[[str], int]:
            def inner(s: str) -> int:
                i = convert(s)
                if i < min_:
                    raise ValueError(f"Too small.")
                return i

            return inner

        self._seed_tkvar = tk.StringVar(value=str(self._seed))
        self._create_input_field(controls_frame, "Seed: ", self._seed_tkvar)
        self._tkvar_bindings.append((self._seed_tkvar, int, "_seed"))

        increment_seed_tkvar = tk.StringVar(value="1" if self._increment_seed else "0")
        self._create_radio_buttons(
            controls_frame, "Increment seed after restart: ", increment_seed_tkvar, ["Yes", "No"], ["1", "0"]
        )
        self._tkvar_bindings.append((increment_seed_tkvar, lambda x: x == "1", "_increment_seed"))

        pop_tkvar = tk.StringVar(value=str(self._pop))
        self._create_input_field(controls_frame, "Population: ", pop_tkvar)
        self._tkvar_bindings.append((pop_tkvar, with_min(int, 1), "_pop"))

        pop_shape_tkvar = tk.StringVar(value=self._pop_shape)
        self._create_radio_buttons(
            controls_frame, "Population shape: ", pop_shape_tkvar, ["Random", "Grid"], ["random", "grid"]
        )
        self._tkvar_bindings.append((pop_shape_tkvar, str, "_pop_shape"))

        maximise_tkvar = tk.StringVar(value="1" if self._maximise else "0")
        self._create_radio_buttons(
            controls_frame, "Optimisation mode: ", maximise_tkvar, ["Minimise", "Maximise"], ["0", "1"]
        )
        self._tkvar_bindings.append((maximise_tkvar, lambda x: x == "1", "_maximise"))

        inertia_tkvar = tk.StringVar(value=str(self._inertia))
        self._create_input_field(controls_frame, "Inertia: ", inertia_tkvar)
        self._tkvar_bindings.append((inertia_tkvar, np.float64, "_inertia"))

        p_coef_tkvar = tk.StringVar(value=str(self._p_coef))
        self._create_input_field(controls_frame, "Personal coefficient: ", p_coef_tkvar)
        self._tkvar_bindings.append((p_coef_tkvar, np.float64, "_p_coef"))

        g_coef_tkvar = tk.StringVar(value=str(self._g_coef))
        self._create_input_field(controls_frame, "Global coefficient: ", g_coef_tkvar)
        self._tkvar_bindings.append((g_coef_tkvar, np.float64, "_g_coef"))

        init_v_tkvar = tk.StringVar(value=str(self._init_v))
        self._create_input_field(controls_frame, "Initial velocity: ", init_v_tkvar)
        self._tkvar_bindings.append((init_v_tkvar, np.float64, "_init_v"))

        tps_tkvar = tk.StringVar(value=str(self._tps))
        self._create_input_field(controls_frame, "Ticks per second: ", tps_tkvar)
        self._tkvar_bindings.append((tps_tkvar, with_min(np.float64, 0), "_tps"))

        spawn_amount_tkvar = tk.StringVar(value=str(self._spawn_amount))
        self._create_input_field(controls_frame, "Spawn amount: ", spawn_amount_tkvar)
        self._tkvar_bindings.append((spawn_amount_tkvar, with_min(int, 0), "_spawn_amount"))

        tk.Label(controls_frame, text="Parameters only update on restart.").pack(side=tk.TOP, expand=False, fill=tk.X)

        self._best_label = tk.Label(controls_frame, text=f"Best value: none yet\n")
        self._best_label.pack(side=tk.TOP, expand=False, fill=tk.X)

        restart_button = tk.Button(controls_frame, text="RESTART SIMULATION", command=self._restart_sim)
        restart_button.pack(side=tk.TOP, expand=True, fill=tk.BOTH)

        # update to resize widgets
        self._window.update()
        self._window.minsize(
            canvas_frame.winfo_width() + controls_frame.winfo_width(),
            max(canvas_frame.winfo_height(), controls_frame.winfo_height()),
        )

    @staticmethod
    def _create_input_field(parent: tk.Widget, name: str, tkvar: tk.StringVar) -> None:
        """Create a labeled input field for a simulation parameter."""
        frame = tk.Frame(parent)
        frame.pack(side=tk.TOP, expand=False, fill=tk.X)
        label = tk.Label(frame, text=name)
        label.pack(side=tk.LEFT, expand=False)
        entry = tk.Entry(frame, textvariable=tkvar)
        entry.pack(side=tk.RIGHT, expand=False)

    @staticmethod
    def _create_radio_buttons(
        parent: tk.Widget, name: str, tkvar: tk.StringVar, button_names: List[str], values: List[str]
    ) -> None:
        """Create a labeled column of radio buttons for a simulation parameter."""
        frame = tk.Frame(parent)
        frame.pack(side=tk.TOP, expand=False, fill=tk.X)
        label = tk.Label(frame, text=name)
        label.pack(side=tk.LEFT, expand=False, fill=tk.X)
        radio_frame = tk.Frame(frame)
        radio_frame.pack(side=tk.RIGHT, expand=False, fill=tk.X)
        for button_name, value in zip(button_names, values):
            button = tk.Radiobutton(radio_frame, text=button_name, variable=tkvar, value=value, anchor=tk.W)
            button.pack(side=tk.TOP, expand=False, fill=tk.X)

    def run(self) -> None:
        """
        Main app entry point.
        Initialises the GUI and starts the tk main event loop.
        """
        self._init_gui()
        # reuse the restart mechanism for initialisation
        self._restart_sim()
        self._window.mainloop()

    def _tick(self) -> None:
        """Run a simulation step, update outputs and schedule the next tick."""
        if self._tps > 0:
            next_tick_time = time.time() + 1 / self._tps
        else:
            next_tick_time = None
        self._sim.step()
        self._draw()
        if next_tick_time is not None:
            # ticks are always at least 1ms apart to keep the GUI somewhat responsive
            next_tick_in = max(int((next_tick_time - time.time()) * 1000), 1)
            self._next_tick = self._window.after(next_tick_in, self._tick)

    def _draw(self) -> None:
        """
        Update particle visualisation & best value display.
        """
        for i, part in enumerate(self._sim.particles):
            if i >= len(self._drawn_particles):
                # Tkinter is able to handle decimal coordinates, but it doesn't round them consistently,
                # resulting in jittery movement. Converting to integers fixes this.
                body = self._canvas.create_oval(
                    int(part.pos[1] - self._particle_radius),
                    int(part.pos[0] - self._particle_radius),
                    int(part.pos[1] + self._particle_radius),
                    int(part.pos[0] + self._particle_radius),
                    outline=self._particle_colour,
                    width=self._line_thickness,
                )
                self._drawn_particles.append(body)
                continue
            body = self._drawn_particles[i]
            # See above
            self._canvas.moveto(
                body, int(part.pos[1] - self._particle_offset), int(part.pos[0] - self._particle_offset)
            )
        best_val, (best_y, best_x) = self._sim.best()
        self._best_label.config(text=f"Best value: {best_val :0.20e}\n at X={best_x :0.10e}, Y={best_y :0.10e}")

    def _clear_canvas(self) -> None:
        """
        Clear the canvas & drawn particle identifier list.
        Used when restarting the simulation.
        """
        self._canvas.delete(*self._drawn_particles)
        self._drawn_particles.clear()

    def _restart_sim(self, _event: Optional[tk.Event] = None) -> None:
        """
        Restart the simulation with parameters from the GUI.
        Invalid parameters will be reset to last valid values.
        """
        self._clear_canvas()
        # read inputs, reset invalid ones
        for tkvar, convert, var in self._tkvar_bindings:
            try:
                setattr(self, var, convert(tkvar.get()))
            except ValueError:
                tkvar.set(str(getattr(self, var)))
        if self._increment_seed:
            self._seed_tkvar.set(str(self._seed + 1))
        if self._next_tick is not None:
            # cancel old tick loop
            self._window.after_cancel(self._next_tick)
            self._next_tick = None
        self._init_sim()
        self._draw()
        if self._tps > 0:
            # start new tick loop
            self._next_tick = self._window.after(int(1000 / self._tps), self._tick)

    def _spawn_particle(self, pos: Optional[npt.NDArray[np.float64]] = None) -> None:
        """Spawn a new particle at `pos` (or randomly if `pos` is `None`)."""
        self._sim.spawn_particle(pos)
        self._draw()

    def _quit(self, _event: Optional[tk.Event] = None) -> None:
        """Handle the window close event by stopping the app."""
        self._window.quit()

    def _spawn_particles_on_click(self, event: tk.Event) -> None:
        """Handle mouse clicks on the canvas by spawning particles."""
        for _ in range(self._spawn_amount):
            self._spawn_particle(np.array([event.y, event.x]))


def main(args: argparse.Namespace) -> None:
    """eeby deeby doo."""
    if not os.path.isfile(args.in_file):
        raise ValueError(f"File {args.in_file} does not exist.")
    if args.input_type is None:
        suffix = args.in_file.split(".")[-1]
        match suffix:
            case "bmp" | "png" | "jpg" | "jpeg" | "tiff" | "tif" | "webp":
                args.input_type = "image"
            case "pkl" | "pickle":
                args.input_type = "pickle"
            case _:
                raise ValueError("Could not infer input type from file extension, please use the --input_type option.")

    print(f"Reading input from {args.in_file}...")
    if args.input_type == "image":
        with PIL.Image.open(args.in_file) as img:
            f = BitmapFun(np.array(img.convert("L")), args.sim_size)
    elif args.input_type == "pickle":
        with open(args.in_file, "rb") as in_file:
            f = dill.load(in_file)(args.sim_size)
    else:
        raise ValueError("Invalid input type.")

    if args.sim_size < 1:
        raise ValueError("Simulation size must be at least 1.")

    vals = try_get_cached_vals(args.in_file, args.sim_size)
    if vals is None:
        print("Function values (for background) not found in cache, computing values...")
        t = time.time()
        vals = compute_vals(f, args.sim_size, args.sim_size)
        print(f"Computing function values took {time.time() - t :0.2f} seconds.")
        cache_vals(args.in_file, vals)

    gradient_low = try_parse_rgb(args.bg_low)
    gradient_high = try_parse_rgb(args.bg_high)
    if gradient_low is None:
        print(f"Could not parse colour: {args.bg_low}, using default.")
        gradient_low = np.array([20, 0, 130], dtype=np.uint8)
    if gradient_high is None:
        print(f"Could not parse colour: {args.bg_high}, using default.")
        gradient_high = np.array([130, 255, 130], dtype=np.uint8)
    print("Rendering background...")
    bg = vals_to_image(vals, gradient_low, gradient_high, args.log_bg)

    print("Initialising the simulation...")
    app = App(
        seed=args.seed,
        f=f,
        maximise=args.maximise,
        pop_type=args.pop_shape,
        pop=args.pop,
        inertia=args.inertia,
        p_coef=args.p_coef,
        g_coef=args.g_coef,
        init_v=args.init_v,
        sim_size=args.sim_size,
        tps=args.tps,
        particle_colour=args.particle_colour,
        particle_radius=args.particle_radius,
        line_thickness=args.line_thickness,
        bg=bg,
        increment_seed=args.increment_seed,
        spawn_amount=args.spawn_amount,
    )
    print("Running...")
    app.run()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file", help="input file with the function to optimise")
    parser.add_argument(
        "--input_type",
        choices=["image", "pickle"],
        help="type of the input file (inferred from file extension if not set)",
    )
    parser.add_argument("--sim_size", type=int, help="size of the simulation area", default=800)
    parser.add_argument("--maximise", action="store_true", help="maximise the function instead of minimising it")
    parser.add_argument("--seed", type=int, help="random seed for the simulation", default=int(time.time()))
    parser.add_argument(
        "--no_increment_seed",
        action="store_false",
        dest="increment_seed",
        help="don't increment the seed after each reset",
    )
    parser.add_argument("--pop", metavar="#N", type=int, help="population size", default=16)
    parser.add_argument(
        "--pop_shape", choices=["random", "grid"], help="initial population distribution shape", default="random"
    )
    parser.add_argument(
        "--spawn_amount", metavar="K", type=int, help="amount of particles to spawn on click", default=1
    )
    parser.add_argument("--inertia", type=np.float64, help="inertia coefficient", default=0.99)
    parser.add_argument("--p_coef", type=np.float64, help="personal coefficient", default=0.01)
    parser.add_argument("--g_coef", type=np.float64, help="global coefficient", default=0.01)
    parser.add_argument("--init_v", type=np.float64, help="initial velocity", default=10)
    parser.add_argument("--tps", type=np.float64, help="ticks per second", default=24)  # cinematic
    parser.add_argument("--particle_colour", metavar="#RRGGBB", help="colour of particle circles", default="0x000000")
    parser.add_argument("--particle_radius", metavar="px", type=int, help="radius of particle circles", default=6)
    parser.add_argument(
        "--line_thickness", metavar="px", type=int, help="base line thickness (for particle outlines etc.)", default=2
    )
    parser.add_argument("--bg_low", metavar="#RRGGBB", help="low end of the background gradient", default="0x221188")
    parser.add_argument("--bg_high", metavar="#RRGGBB", help="high end of the background gradient", default="0xaaffaa")
    parser.add_argument("--log_bg", action="store_true", help="use logarithmic scale for the background gradient")
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
