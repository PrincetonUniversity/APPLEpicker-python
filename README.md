# apple-picker-python

This is the Python version of Apple-Picker project.
If you are looking for the MATLAB version, you can find it [Here](https://github.com/PrincetonUniversity/APPLEpicker-python).

Make sure to [subscribe](http://eepurl.com/dFmFfn) to important updates, tips and tricks about Apple-Picker.

## Installation Process (Linux, Debian based OS)
First, verify your machine has Python3.5 (and up) installed by running: `Python3 -V`

Then, make sure Pip package manager for Python is installed. Run `apt install -y python3-pip --reinstall`

As a next step, install Pipenv to easily manage Virtualenv and dependencies: `pip3 install pipenv setuptools`

If you haven't, clone the project to your local machine:

`git clone https://github.com/PrincetonUniversity/APPLEpicker-python.git`

Move into the root folder of the project: `cd APPLEpicker-python`, you should be able to see the file `apple.py` if you run `ls`.

Now, create a virtual environment for Apple-Picker: `pipenv --python 3.5`

Install all dependencies: `pipenv install`

You're all set! You can now run Apple-Picker within your newly created virtualenv:

`pipenv run python3 apple.py -s particle_size my_mrc_folder`

The ourput files will be placed in a new folder called `star_dir` next to `my_mrc_folder`. You can also specify another output directory using flag `-o output_dir`.

You can override more default values by editing file `config.py`

### Troubleshooting
While running `pipenv install`, you might run into problems because of missing Linux packages.

Try to install the following packages and then run `pipenv install` again:

`apt-get install -y libfftw3-3 libfftw3-bin libfftw3-dbg libfftw3-dev libfftw3-doc libfftw3-double3 libfftw3-long3 libfftw3-mpi-dev libfftw3-mpi3 libfftw3-quad3 libfftw3-single3`
