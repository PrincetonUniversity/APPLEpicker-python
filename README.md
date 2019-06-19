# APPLEpicker-python
## Automatic Particle Picking, a Low-Effort Cryo-EM Framework

This is the Python version of APPLE picker, a particle picker for single-particle cryo-electron microscopy (cryo-EM).
For more information on the algorithm, please see the [paper](https://doi.org/10.1016/j.jsb.2018.08.012).

If you are looking for the MATLAB version, please see the [APPLEpicker page](https://github.com/PrincetonUniversity/APPLEpicker).

Make sure to [subscribe](http://eepurl.com/dFmFfn) to important updates, tips and tricks about the APPLE picker.

## Installation Instructions (Linux/Mac OS X/Windows)

Install [Anaconda](https://www.anaconda.com/distribution/) 64-bit for your platform.

Next, run

```
git clone https://github.com/PrincetonUniversity/APPLEpicker-python.git
cd /path/to/git/clone/folder
conda env create -f environment.yml
```

You're all set! You can now run the APPLE-Picker within your newly created applepicker enviroment:

```
conda activate applepicker
python /path/to/git/clone/folder/run.py -s particle_size mrc_dir
```

where `particle_size` is the expected size of the particles in pixels and `mrc_dir` is the folder containing the micrographs. The output files will be placed in a new folder called `star_dir` in the same parent folder as `mrc_dir`. You can also specify another output directory using flag `-o output_dir`.

If you want APPLE-Picker to create a JPG image of the result, pass the flag `--jpg`.

To see help text, simply pass `-h` or `--help`.

You can override more default values by editing the file `config.py`

