# Simple PSO visualiser

The PSO algorithm optimises a scalar function by simulating a "swarm" of particles with local and global memory.  
This little program visualises the algorithm for functions on 2d square domains.  
It's not really meant for actually optimising functions so don't use it for that. 


## Requirements

- Python 3.11 (with `tkinter`)
- Python libraries: `dill (0.3)`, `numpy (1.26)`, `pillow (10.2)` (you can install all with `pip install -r requirements.txt`)
- Tk and Tcl


## Usage

***The program will create directories & files in the current working directory.***

The simplest way to laun\ch the program is `python app.py INPUT_FILE`. `INPUT_FILE` is the only required argument, a path to an input file. 
Input files can be either images (any format the Pillow library can read) or `dill`-pickled functions that take the domain size as a parameter and return the actual function mapping coordinates to a real value.

For more options see below or `python app.py -h`.


### Command line parameters

- `--input_type` allows you to manually specify the input file type 
(in case the extension doesn't get recognised or doesn't match)
- `--sim_size` specifies the side length (in pixels) of the simulation area.


#### Simulation parameters

- `--maximise` switches to maximising the function instead of minimising
- `--seed` specifies the seed for the simulation. 
Keeping it the same (along with other parameters) should result in identical runs.
- `--no_increment_seed` prevents the seed from being incremented after each reset.
- `--pop` specifies the initial population size
- `--pop_shape` determines the initial distribution of the population: 
`random` is uniformly random, 
`grid` is uniformly distributed in a grid. 
(With `grid` the population size is rounded up to the nearest integer square.)
- `--spawn_amount` controls how many new particles get spawned when you click the simulation area.
- `--inertia` determines how much of their velocity the particles maintain between ticks.
- `--p_coef` affects how strongly a particle accelerates towards its personal best.
- `--g_coef` affects how strongly a particle accelerates towards the global best.
- `--init_v` controls the maximum initial velocity of each particle. 
(In pixels / units per tick; the initial velocity in each direction is independently chosen from U(-init_v,init_v))
- `--tps` controls the amount of ticks per second. Setting it to 0 will cause the simulation not to run.


#### GUI parameters

These can only be set using command line arguments.
- `--particle_colour` determines the colour of the particle circles. 
Like with other colours, the program expects an 8bit RGB hex code with or without an `#` and `0x` at the start 
(that is, anything matching `#?(0[Xx])?[0-9A-Fa-f]{6}` should work.)
- `--particle_radius` is the radius of the circles in pixels
- `--line_thickness` is the (roughly) thickness of the circles in pixels. 
It's also used as a base for computing thickness of borders 
(thin borders didn't look good with thick circles)
- `--bg_low` and `--bg_high` are the colours (same format as above) for the background gradient 
(`--bg_low` being used for the lowest sampled value, `--bg_high` for the highest).
- `--log_bg` if set, makes the background gradient log scaled. 
Useful to highlight minima, but doesn't interact well with `--maximise`


### The GUI

#### What the GUI displays

In the foreground, a small (at least with default settings) circle for each particle.  
In the background, a gradient roughly visualising the function to optimise. 
The gradient is produced by sampling at each pixel, so it might not be entirely accurate for very noisy functions. 
The sampled values are then log scaled if `--log_bg` is set, min-max scaled and used as coefficients to create a gradient 
by blending 2 colours in the OKLab colour space (see the Internals section for explanation). 
If there are infinities present among the sampled values, they will be offset from the finite values by a bit.


#### Interacting with the GUI

You can set *simulation* parameters in the GUI using input fields and radio buttons. 
The values only update on reset and invalid inputs are reset to previous values.  
You can spawn additional particles by clicking in the visualisation area 
(Try starting with a single particle and adding more one by one, it's fun!)
Besides the big reset button, it's also possible to reset the simulation by pressing ctrl-r.  
You can quit the program by closing the window or by pressing ctrl-q.


### Demos & helper scripts

Running `demo_functions.py` will prepare some pickled functions, `demo_images.py` will download some images. 
Both scripts put their outputs in `./demos`. 
You can also run `prepare_all_demos.py` which will prepare the demos as well as precompute function values for background rendering (for 800x800 mode).  
Running `clear_cache.py` simply deletes the whole `.cache` directory..

---

## Internals

- `app.py` is the main entrypoint as well as containing the definition of the `App` class.
- `pso.py` contains the implementation of the actual PSO algorithm.
- `fileutil.py` contains utility functions for dealing with caching and saving files.
- `imgutil.py` contains functions that interpret and create images.
- `demo_images.py`, `demo_functions.py`, `prepare_all_demos.py` and `clear_cache.py` are helper scripts. 

Now to go over the at least slightly tricky bits:


#### Inputs

**Image input** works by loading the image using Pillow and converting it to greyscale; function values are then computed by translating coordinates inside the search space into coordinates inside the image, then doing bilinear interpolation (weighted sum of the closest 4 pixels.)  
**Pickle input** uses `dill` because standard library `pickle` doesn't support pickling functions. It's essentially just glorified source files and modules used in the functions must also be loaded in `app.py`, but it was simpler than trying to figure out some custom way to save and load source code.


#### The tick loop

I initially handled GUI events by calling `Tk.update()` inside a loop that ran once per tick. That felt unresponsive at low (single digit) tickrates and so I decided to switch to using `Tk.mainloop()` with ticks being scheduled using `Tk.after()`. This works mostly fine, and with some precautions to make sure ticks are scheduled at least 1 ms in the future it does not perform worse than the old way. (That is, it struggles very similarly under high load.)


#### Caching values for backgrounds

The gradient background requires sampling the function *size* x *size* times, which can take a relatively long time. I don't think there is a way to make it faster without imposing otherwise unneccessary requirements on the function inputs, but 10+ second startup times aren't exactly acceptable. (They were especially annoying during development when there was no way to restart the simulation without restarting the whole program.)  
For this reason, the program caches the values. When loading an input it first checks `./.cache/hashes` for a hash file for the input file name (not the whole path), compares it against the file's hash if present, then checks `./.cache/vals` for a precomputed matrix of desired size. If not found, it is computed the slow way, then cached to avoid having to compute it again as long as the input doesn't change.


#### Background gradients

RGB gradients are ugly. HSV gradients are almost as bad, and although HLS was giving better results than the other two, I was getting hard transitions after converting back to RGB that I couldn't get rid of.  
I decided to switch to [OKLab](https://bottosson.github.io/posts/oklab/) colors, as those were pretty much made for making pretty gradients. The conversion code contained in `imgutil.py` is adapted from the linked blogpost.
