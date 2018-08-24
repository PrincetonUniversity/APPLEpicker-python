#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 09:40:13 2018

@author: Ayelet Heimowitz, Itay Sason
"""
import argparse
import glob
import numpy as np
import os

from functools import partial
from multiprocessing import Pool

from picking import Picker
from config import ApplePickerConfig


class Apple(object):

    def __init__(self, config):

        self.particle_size = config.particle_size
        self.query_image_size = config.query_image_size
        self.query_window_size = config.query_window_size
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
            self.max_particle_size = self.particle_size * 4

        if self.min_particle_size is None:
            self.min_particle_size = int(self.particle_size / 4)

        if self.minimum_overlap_amount is None:
            self.minimum_overlap_amount = int(self.particle_size / 10)

        qBox = (4000 ** 2) / (self.query_image_size ** 2) * 4
        if self.tau1 is None:
            self.tau1 = int(qBox * 3 / 100)

        if self.tau2 is None:
            self.tau2 = int(qBox * 30 / 100)
            
        self.verify_input_values()

    def verify_input_values(self):
        # TODO
        return
        # if self.maxSize < 1:
        #     messagebox.showerror("Error", "Max particle size must be a positive integer.")
        #     return 1
        # if self.maxSize > 3000:
        #     messagebox.showerror("Error", "Max particle size is too large.")
        #     return 1
        #
        # if self.qSize < 1:
        #     messagebox.showerror("Error", "Query image size must be a positive integer.")
        #     return 1
        # if self.qSize > 3000:
        #     messagebox.showerror("Error", "Query image size is too large.")
        #     return 1
        #
        # if self.pSize < 5:
        #     messagebox.showerror("Error", "Particle size too small.")
        #     return 1
        # if self.pSize > 3000:
        #     messagebox.showerror("Error", "Particle size too large.")
        #     return 1
        #
        # if self.minSize < 1:
        #     messagebox.showerror("Error", "Min particle size must be a positive integer.")
        #     return 1
        # if self.minSize > 3000:
        #     messagebox.showerror("Error", "Min particle size is too large.")
        #     return 1
        #
        # if self.tau1 < 0:
        #     messagebox.showerror("Error", "\u03C4\u2081 must be a positive integer.")
        #     return 1
        # if self.tau1 > (4000 / self.qSize * 2) ** 2:
        #     messagebox.showerror("Error", "\u03C4\u2081 is too large.")
        #     return 1
        #
        # if self.tau2 < 0:
        #     messagebox.showerror("Error", "\u03C4\u2082 must be a positive integer.")
        #     return 1
        # if self.tau2 > (4000 / self.qSize * 2) ** 2:
        #     messagebox.showerror("Error", "\u03C4\u2082 is too large.")
        #     return 1
        #
        # if self.moa < 1:
        #     messagebox.showerror("Error", "Min overlap must be a positive integer.")
        #     return 1
        # if self.moa > 3000:
        #     messagebox.showerror("Error", "Min overlap is too large.")
        #     return 1
        #
        # if self.cSize * 2 + 200 > 4000:
        #     messagebox.showerror("Error", "Container size is too big.")
        #     return 1
        # if self.cSize < self.pSize:
        #     messagebox.showerror("Error", "Container size must exceed particle size.")
        #     return 1
        # if self.pSize < self.qSize:
        #     messagebox.showerror("Error", "Particle size must exceed query image size.")
        #     return 1
        # if self.dir_name_in == '':
        #     messagebox.showerror("Error", "No input directory selected.")
        #     return 1
        # if self.dir_name_out == '':
        #     messagebox.showerror("Error", "No output directory selected.")
        #     return 1
        # if self.proc < 1:
        #     messagebox.showerror("Error", "Please select at least one processor.")
        #     return 1
        # return 0

    def pick_particles(self, mrc_dir):

        # fetch all mrc files from mrc folder
        filenames = [os.path.basename(file) for file in glob.glob(f'{mrc_dir}/*.mrc')]
        print("converting mrc files:", filenames)

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
        data.append(self.output_dir if self.output_dir else mrc_dir)

        pool = Pool(processes=self.proc)
        partial_func = partial(Apple.process_micrograph, data)
        pool.map(partial_func, filenames)
        pool.terminate()

    def process_micrograph(data, filenames):
        file_basename, file_extension = os.path.splitext(filenames)

        # parse filename and verify extension is ".mrc"
        if file_extension == '.mrc':  # todo use negative condition, use other func

            picker = Picker()

            directory = data[0]
            pSize = data[1]
            maxSize = data[2]
            minSize = data[3]
            qSize = data[4]
            tau1 = data[5]
            tau2 = data[6]
            moa = data[7]
            cSize = data[8]
            directory_out = data[9]

            # add path to filename
            filename = directory + '/' + filenames

            # Initialize parameters for the APPLE picker
            Picker.initializeParameters(picker, pSize, maxSize, minSize, qSize, tau1, tau2, moa, cSize, filename,
                                        directory_out)

            # update user
            print('Processing ', end='')
            nameParse = filenames.split("/")
            print(nameParse[-1])

            # return .mrc file as a float64 array
            microImg = Picker.readMRC(picker)  # return a micrograph as an numpy array

            # bin and filter micrograph
            #            microImg = picking.initialManipulations(picker, microImg) # filtering, binning, etc.

            # compute score for query images
            score = Picker.queryScore(picker, microImg)  # compute score using normalized cross-correlations
            flag = True

            while (flag):
                # train SVM classifier and classify all windows in micrograph
                segmentation = Picker.runSVM(picker, microImg, score)

                # If all windows are classified identically, update tau_1 or tau_2
                if np.array_equal(segmentation, np.ones((segmentation.shape[0], segmentation.shape[1]))):
                    tau2 = tau2 + 1
                elif np.array_equal(segmentation, np.zeros((segmentation.shape[0], segmentation.shape[1]))):
                    tau1 = tau1 + 1
                else:
                    flag = False

            # discard suspected artifacts
            segmentation = Picker.morphologyOps(picker, segmentation)

            # create output star file
            Picker.extractParticles(picker, segmentation)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Apple Picker')
    parser.add_argument("-s", type=int, metavar='my_particle_size', help="size of particle")
    parser.add_argument("-o", type=str, metavar="output dir",
                        help="name of output folder where (by default AP saves to input folder "
                             "and adds 'picked' to original file name.)")

    parser.add_argument("mrcdir", metavar='input dir', type=str,
                        help="path to folder containing all mrc files to pick.")

    args = parser.parse_args()

    if args.s:
        ApplePickerConfig.particle_size = args.s

    if ApplePickerConfig.particle_size is None:
        raise Exception("particle size is not defined! either set it with -s or adjust config.py")

    if args.o:
        ApplePickerConfig.output_dir = args.o

    apple = Apple(ApplePickerConfig)
    apple.pick_particles(args.mrcdir)
