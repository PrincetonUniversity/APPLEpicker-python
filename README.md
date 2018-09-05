# APPLEpicker-python
## Automatic Particle Picking, a Low-Effort Cryo-EM Framework

This is the Python version of APPLE picker, a particle picker for single-particle cryo-electron microscopy (cryo-EM).
For more information on the algorithm, please see the [paper](https://arxiv.org/abs/1802.00469).

If you are looking for the MATLAB version, please see the [APPLEpicker page](https://github.com/PrincetonUniversity/APPLEpicker).

Make sure to [subscribe](http://eepurl.com/dFmFfn) to important updates, tips and tricks about Apple-Picker.

## Installation Process (Linux, Debian based OS)
First, verify that your machine has Python 3.5 (or newer) installed by running: `python3 -V`

Then make sure Pip package manager for Python is installed. Run `apt install -y python3-pip --reinstall`

As a next step, install Pipenv to easily manage Virtualenv and dependencies: `pip3 install pipenv setuptools`

If you haven't already, clone the project to your local machine:

`git clone https://github.com/PrincetonUniversity/APPLEpicker-python.git`

Move into the root folder of the project: `cd APPLEpicker-python`. You should be able to see the file `apple.py` if you run `ls`.

Now create a virtual environment for Apple-Picker: `pipenv --python 3.5` (or whatever Python version you have).

Install all dependencies: `pipenv install`.

You're all set! You can now run Apple-Picker within your newly created virtualenv:

`pipenv run python3 apple.py -s particle_size mrc_dir`

where `particle_size` is the expected size of the particles in pixels and `mrc_dir` is the folder containing the micrographs. The ourput files will be placed in a new folder called `star_dir` in the same parent folder as `mrc_dir`. You can also specify another output directory using flag `-o output_dir`.

If you want Apple-Picker to create a JPG image of the result, pass the flag `--jpg`.

To see help text, simply pass `-h` or `--help`.

You can override more default values by editing the file `config.py`

### Troubleshooting
While running `pipenv install`, you might run into problems because of missing Linux packages.

Try to install the missing packages using

`apt-get install -y libfftw3-3 libfftw3-bin libfftw3-dbg libfftw3-dev libfftw3-doc libfftw3-double3 libfftw3-long3 libfftw3-mpi-dev libfftw3-mpi3 libfftw3-quad3 libfftw3-single3`

and then run `pipenv install` again.
