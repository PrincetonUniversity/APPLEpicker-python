import argparse
import os
import sys

from apple.apple import Apple
from apple.config import ApplePickerConfig
from apple.exceptions import ConfigError

# fix future warning with a patch
import apple.scipy_signaling_patch  # pylint: disable=unused-import


parser = argparse.ArgumentParser(description='Apple Picker')
parser.add_argument("-s", type=int, metavar='my_particle_size', help="size of particle")
parser.add_argument("--jpg", action='store_true', help="create result image")
parser.add_argument("-o", type=str, metavar="output dir",
                    help="name of output folder where star file should be saved (by default "
                         "AP saves to input folder and adds 'picked' to original file name.)")

parser.add_argument("mrcdir", metavar='input dir', type=str,
                    help="path to folder containing all mrc files to pick.")

args = parser.parse_args()

if args.s:
    ApplePickerConfig.particle_size = args.s

if ApplePickerConfig.particle_size is None:
    raise Exception("particle size is not defined! either set it with -s or adjust config.py")

if args.o:
    if not os.path.exists(args.o):
        raise ConfigError("Output directory doesn't exist! {}".format(args.o))
    ApplePickerConfig.output_dir = args.o

if args.jpg:
    ApplePickerConfig.create_jpg = True


if not os.path.exists(args.mrcdir):
    print("mrc folder {} doesn't' exist! terminating..".format(args.mrcdir))
    sys.exit(1)

if not os.listdir(args.mrcdir):
    print("mrc folder is empty! terminating..")
    sys.exit(1)

apple = Apple(ApplePickerConfig, args.mrcdir)
apple.pick_particles()
