#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 09:40:13 2018

@author: Ayelet Heimowitz, Itay Sason
"""
import argparse
import glob
import os
import numpy as np

from functools import partial
from multiprocessing import Pool

from exceptions import ConfigError
from picking import Picker
from config import ApplePickerConfig


class Apple:

    def __init__(self, config, mrc_dir):

        self.particle_size = config.particle_size
        self.query_image_size = config.query_image_size
        self.max_particle_size = config.max_particle_size
        self.min_particle_size = config.min_particle_size
        self.minimum_overlap_amount = config.minimum_overlap_amount
        self.tau1 = config.tau1
        self.tau2 = config.tau2
        self.container_size = config.container_size
        self.proc = config.proc
        self.output_dir = config.output_dir

        # set default values if needed
        query_window_size = np.round(self.particle_size * 2 / 3)
        query_window_size -= query_window_size % 4
        query_window_size = int(query_window_size)

        self.query_image_size = query_window_size

        if self.max_particle_size is None:
            self.max_particle_size = self.particle_size * 2

        if self.min_particle_size is None:
            self.min_particle_size = int(self.particle_size / 4)

        if self.minimum_overlap_amount is None:
            self.minimum_overlap_amount = int(self.particle_size / 10)

        if self.output_dir is None:
            path = os.path.dirname(mrc_dir)
#            abs_path = os.path.abspath(mrc_dir)
#            self.output_dir = abs_path.replace(path, 'star_dir')
            self.output_dir = os.path.join(path, 'star_dir')
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

        q_box = (4000 ** 2) / (self.query_image_size ** 2) * 4
        if self.tau1 is None:
            self.tau1 = int(q_box * 3 / 100)

        if self.tau2 is None:
            self.tau2 = int(q_box * 30 / 100)

        self.verify_input_values()
        self.print_values()

    def print_values(self):
        """Printing all parameters to screen."""
        
         try:
            std_out_width = os.get_terminal_size().columns
        except OSError:
            std_out_width = 100
        
        print(' Parameter Report '.center(std_out_width, '=') + '\n')

        params = ['particle_size',
                  'query_image_size',
                  'max_particle_size',
                  'min_particle_size',
                  'minimum_overlap_amount',
                  'tau1',
                  'tau2',
                  'container_size',
                  'proc',
                  'output_dir']

        for param in params:
            print('%(param)-40s %(value)-10s' % {"param": param, "value": getattr(self, param)})

        print('\n' + ' Progress Report '.center(std_out_width, '=') + '\n')

    def verify_input_values(self):
        """Verify parameter values make sense.
        
        Sanity check for the attributes of this instanse of the Apple class.
        
        Raises:
            ConfigError: Attribute is out of range.
        """
        
        if not 1 <= self.max_particle_size <= 3000:
            raise ConfigError("Error", "Max particle size must be in range [1, 3000]!")

        if not 1 <= self.query_image_size <= 3000:
            raise ConfigError("Error", "Query image size must be in range [1, 3000]!")

        if not 5 <= self.particle_size < 3000:
            raise ConfigError("Error", "Particle size must be in range [5, 3000]!")

        if not 1 <= self.min_particle_size < 3000:
            raise ConfigError("Error", "Min particle size must be in range [1, 3000]!")

        max_tau1_value = (4000 / self.query_image_size * 2) ** 2
        if not 0 <= self.tau1 <= max_tau1_value:
            raise ConfigError("Error",
                              "\u03C4\u2081 must be a in range [0, {}]!".format(max_tau1_value))

        max_tau2_value = (4000 / self.query_image_size * 2) ** 2
        if not 0 <= self.tau2 <= max_tau2_value:
            raise ConfigError("Error",
                              "\u03C4\u2082 must be in range [0, {}]!".format(max_tau2_value))

        if not 0 <= self.minimum_overlap_amount <= 3000:
            raise ConfigError("Error", "overlap must be in range [0, 3000]!")

        # max container_size condition is (conainter_size_max * 2 + 200 > 4000), which is 1900
        if not self.particle_size <= self.container_size <= 1900:
            raise ConfigError("Error", "Container size must be within range [{}, 1900]!".format(
                self.particle_size))

        if self.particle_size < self.query_image_size:
            raise ConfigError("Error",
                              "Particle size must exceed query image size! particle size:{}, "
                              "query image size: {}".format(self.particle_size,
                                                            self.query_image_size))

        if self.proc < 1:
            raise ConfigError("Error", "Please select at least one processor!")

    def pick_particles(self, mrc_dir):
        """Initiate picking.
        
        Creates a pool of processes and initializes the process of particle picking on each micrograph. 
        A single process is used per micrograph.
        
        Args:
            mrc_dir: Directory containing the micrographs to be picked.
        """

        # fetch all mrc files from mrc folder
        filenames = [os.path.basename(file) for file in glob.glob('{}/*.mrc'.format(mrc_dir))]
        print("converting {} mrc files..".format(len(filenames)))

        data = list()
        data.append(mrc_dir)
        data.append(self.particle_size)
        data.append(self.max_particle_size)
        data.append(self.min_particle_size)
        data.append(self.query_image_size)
        data.append(self.tau1)
        data.append(self.tau2)
        data.append(self.minimum_overlap_amount)
        data.append(self.container_size)
        data.append(self.output_dir)
        data.append(1)
        
        Apple.process_micrograph(data, filenames[0])
        
        data[10] = 0
        filenames.remove(filenames[0])
        
        pool = Pool(processes=self.proc)
        partial_func = partial(Apple.process_micrograph, data)
        pool.map(partial_func, filenames)
        pool.terminate()

    @staticmethod
    def process_micrograph(data, filename):
        """Pick particles.
        
        Implemets the APPLE picker algorithm (Heimowitz, Andén and Singer, 
        "APPLE picker: Automatic particle picking, a low-effort cryo-EM framework").
        
        Args:
            data: list of parameters needed for the APPLE picking process.
            filename: Name of micrograph for picking.
            
            Raises:
                ConfigError: Incorrect format for micrograph file.
        """

        if not filename.endswith('.mrc'):
            raise ConfigError("Input file doesn't seem to be an MRC format! ({})".format(filename))

        input_dir = data[0]
        p_size = data[1]
        max_size = data[2]
        min_size = data[3]
        q_size = data[4]
        tau1 = data[5]
        tau2 = data[6]
        moa = data[7]
        c_size = data[8]
        output_dir = data[9]
        show_image = data[10]

        # add path to filename
        filename = os.path.join(input_dir, filename)

        picker = Picker(p_size, max_size, min_size, q_size, tau1, tau2, moa, c_size, filename,
                        output_dir)

        # update user
        print('Processing {}..'.format(os.path.basename(filename)))

        # return .mrc file as a float64 array
        micro_img = picker.read_mrc()  # return a micrograph as an numpy array

        # compute score for query images
        score = picker.query_score(micro_img)  # compute score using normalized cross-correlations

        while True:
            # train SVM classifier and classify all windows in micrograph
            segmentation = picker.run_svm(micro_img, score)

            # If all windows are classified identically, update tau_1 or tau_2
            if np.array_equal(segmentation,
                              np.ones((segmentation.shape[0], segmentation.shape[1]))):
                tau2 = tau2 + 500

            elif np.array_equal(segmentation,
                                np.zeros((segmentation.shape[0], segmentation.shape[1]))):
                tau1 = tau1 + 500

            else:
                break

        # discard suspected artifacts
        segmentation = picker.morphology_ops(segmentation)

        # create output star file
        centers = picker.extract_particles(segmentation)
        
        if (show_image==1):
            picker.display_picks(centers)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Apple Picker')
    parser.add_argument("-s", type=int, metavar='my_particle_size', help="size of particle")
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

    apple = Apple(ApplePickerConfig, args.mrcdir)
    apple.pick_particles(args.mrcdir)
