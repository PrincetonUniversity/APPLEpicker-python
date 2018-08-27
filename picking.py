# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 12:21:28 2018

@author: Ayelet Heimowitz, Itay Sason, Joakim Anden
"""

import mrcfile
import numpy as np
import pyfftw

from scipy import ndimage, misc, signal
from helper import ApplePickerHelper
from sklearn import svm, preprocessing


class Picker(object):

    def __init__(self, particle_size, max_size, min_size, query_size, tau1, tau2, moa, container_size,
                 filenames, output_directory):
        self.particle_size = int(particle_size / 2)
        self.max_size = int(max_size / 4)
        self.min_size = int(min_size / 4)
        self.query_size = int(query_size / 2)
        self.query_size -= self.query_size % 2
        self.tau1 = tau1
        self.tau2 = tau2
        self.MOA = int(moa / 2)
        self.container_size = int(container_size / 2)
        self.filenames = filenames
        self.output_directory = output_directory
        
        self.query_size -= self.query_size % 2

    def read_mrc(self):
        with mrcfile.open(self.filenames, mode='r+', permissive=True) as mrc:
            micro_img = mrc.data

        micro_img = micro_img.astype('float')
        micro_img = micro_img[99:-100, 99:-100]
        micro_img = misc.imresize(micro_img, 0.5, mode='F', interp='cubic')

        gauss_filt = ApplePickerHelper.gaussian_filter(15, 0.5)
        micro_img = signal.correlate(micro_img, gauss_filt, 'same')
        
        micro_img = np.double(micro_img)
        return micro_img
    
    def query_score(self, micro_img):
            
        query_box = ApplePickerHelper.extract_query(micro_img, int(self.query_size / 2))

        out_shape = (query_box.shape[0], query_box.shape[1], query_box.shape[2], query_box.shape[3] // 2 + 1)
        query_box_a = np.empty(out_shape, dtype='complex128')
        fft_class_f = pyfftw.FFTW(query_box, query_box_a, axes=(2, 3), direction='FFTW_FORWARD')
        fft_class_f(query_box, query_box_a)
        query_box = np.conj(query_box_a)

        reference_box_a = ApplePickerHelper.extract_references(micro_img, self.query_size, self.container_size)
        out_shape2 = (reference_box_a.shape[0], reference_box_a.shape[1], reference_box_a.shape[-1] // 2 + 1)
        
        reference_box = np.empty(out_shape2, dtype='complex128')
        fft_class_f2 = pyfftw.FFTW(reference_box_a, reference_box, axes=(1, 2), direction='FFTW_FORWARD')
        fft_class_f2(reference_box_a, reference_box)
        
        conv_map = np.zeros((reference_box.shape[0], query_box.shape[0], query_box.shape[1]))
        
        window_t = np.empty(query_box.shape, dtype=query_box.dtype)
        cc = np.empty((query_box.shape[0], query_box.shape[1], query_box.shape[2], 2*query_box.shape[3]-2),
                      dtype=micro_img.dtype)
        fft_class = pyfftw.FFTW(window_t, cc, axes=(2, 3), direction='FFTW_BACKWARD')

        for index in range(0, reference_box.shape[0]):
            np.multiply(reference_box[index], query_box, out=window_t)
            fft_class(window_t, cc)   
            conv_map[index] = cc.real.max((2, 3)) - cc.real.mean((2,3))
        
        conv_map = np.transpose(conv_map, (1, 2, 0))

        min_val = np.amin(conv_map)
        max_val = np.amax(conv_map)
        thresh = min_val + (max_val - min_val) / 20
        
        h = conv_map >= thresh
        score = np.sum(h, axis=2)

        return score

    def run_svm(self, micro_img, score):
        particle_windows = np.floor(self.tau1)
        non_noise_windows = np.ceil(self.tau2)
        bw_mask_p, bw_mask_n = Picker.get_maps(self, score, micro_img, particle_windows, non_noise_windows)
 
        x, y = ApplePickerHelper.get_training_set(micro_img, bw_mask_p, bw_mask_n, self.query_size)

        scaler = preprocessing.StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)
        classify = svm.SVC(C=1, kernel='rbf', gamma=0.5, class_weight='balanced')
        classify.fit(x, y)                                                  # train SVM classifier
        
        mean_all, std_all = ApplePickerHelper.moments(micro_img, self.query_size)
                                                                    
        mean_all = mean_all[self.query_size - 1:-(self.query_size - 1), self.query_size - 1:-(self.query_size - 1)]
        std_all = std_all[self.query_size - 1:-(self.query_size - 1), self.query_size - 1:-(self.query_size - 1)]

        mean_all = np.reshape(mean_all, (np.prod(mean_all.shape), 1), 'F')
        std_all = np.reshape(std_all, (np.prod(std_all.shape), 1), 'F')
        cls_input = np.concatenate((mean_all, std_all), axis=1)
        cls_input = scaler.transform(cls_input)

        # compute classification for all possible windows in micrograph
        segmentation = classify.predict(cls_input)

        segmentation = np.reshape(segmentation, (int(np.sqrt(segmentation.shape[0])), int(np.sqrt(segmentation.shape[0]))))
        segmentation = segmentation.copy()
        return segmentation
    
    def morphology_ops(self, segmentation):
        if (ndimage.morphology.binary_fill_holes(segmentation)==np.ones(segmentation.shape)).all():
            segmentation[0:100, 0:100] = np.zeros((100,100))
            
        segmentation = ndimage.morphology.binary_fill_holes(segmentation)
        y, x = np.ogrid[-self.min_size:self.min_size+1, -self.min_size:self.min_size+1]
        element = x*x+y*y <= self.min_size * self.min_size
        segmentation_e = ndimage.morphology.binary_erosion(segmentation, element) 
        
        y, x = np.ogrid[-self.max_size:self.max_size+1, -self.max_size:self.max_size+1]
        element = x*x+y*y <= self.max_size * self.max_size
        segmentation_o = ndimage.morphology.binary_erosion(segmentation, element)
        segmentation_o = np.reshape(segmentation_o, (segmentation_o.shape[0], segmentation_o.shape[1], 1)) 
        
        size_const, num_features = ndimage.label(segmentation_e, np.ones((3,3)))
        size_const = np.reshape(size_const, (size_const.shape[0], size_const.shape[1], 1))
        labels = np.unique(size_const*segmentation_o)
        idx = np.where(labels!=0)
        labels = np.take(labels, idx)
        labels = np.reshape(labels, (1,1,np.prod(labels.shape)))
        
        matrix1 = np.repeat(size_const, labels.shape[2], 2)
        matrix2 = np.repeat(labels, matrix1.shape[0], 0)
        matrix2 = np.repeat(matrix2, matrix1.shape[1], 1)
        
        matrix3 = np.equal(matrix1, matrix2)
        matrix4 = np.sum(matrix3, 2)
        
        segmentation_e[np.where(matrix4==1)] = 0
        
        return segmentation_e
    
    def extract_particles(self, segmentation):
        segmentation = segmentation[self.query_size // 2 - 1:-self.query_size // 2,
                       self.query_size // 2 - 1:-self.query_size // 2]
        labeled_segments, num_features = ndimage.label(segmentation, np.ones((3,3)))
        values, repeats = np.unique(labeled_segments, return_counts=True)
        
        values_to_remove = np.where(repeats > self.query_size ** 2)
        values = np.take(values, values_to_remove)
        values = np.reshape(values, (1, 1, np.prod(values.shape)))
        
        labeled_segments = np.reshape(labeled_segments, (labeled_segments.shape[0], labeled_segments.shape[1], 1))
        matrix1 = np.repeat(labeled_segments, values.shape[2], 2)
        matrix2 = np.repeat(values, matrix1.shape[0], 0)
        matrix2 = np.repeat(matrix2, matrix1.shape[1], 1)
        
        matrix3 = np.equal(matrix1, matrix2)
        matrix4 = np.sum(matrix3, 2)
        
        segmentation[np.where(matrix4==1)] = 0
        labeled_segments, num_features = ndimage.label(segmentation, np.ones((3,3)))
        
        max_val = np.amax(np.reshape(labeled_segments, (np.prod(labeled_segments.shape))))
        center = ndimage.measurements.center_of_mass(segmentation, labeled_segments, np.arange(1, max_val))
        center = np.rint(center)
        
        img = np.zeros((segmentation.shape[0], segmentation.shape[1]))
        img[center[:, 0].astype(int), center[:, 1].astype(int)] = 1
        y, x = np.ogrid[-self.MOA:self.MOA+1, -self.MOA:self.MOA+1]
        element = x*x+y*y <= self.MOA*self.MOA
        img = ndimage.morphology.binary_dilation(img, structure=element)
        labeled_img, num_features = ndimage.label(img, np.ones((3,3)))
        values, repeats = np.unique(labeled_img, return_counts=True)
        y = np.where(repeats==np.count_nonzero(element))
        y = np.array(y)
        y = y.astype(int)
        y = np.reshape(y, (np.prod(y.shape)))
        y = y - 1
        center = center[y, :]
        
        center = center + (self.query_size // 2 - 1) * np.ones(center.shape)
        center = center + (self.query_size // 2 - 1) * np.ones(center.shape)
        center = center + np.ones(center.shape)
        center = 2*center
        center = center + 99*np.ones(center.shape)
        
        name_list = self.filenames.split("/")
        name = name_list[-1].split(".")
        name_str = name[0]

        with open(self.output_directory + '/' + name_str +"_applepick.star", "w+") as f:
            np.savetxt(f, ["data_root\n\nloop_\n_rlnCoordinateY #1\n_rlnCoordinateX #2"], fmt='%s')
            np.savetxt(f, center, fmt='%d %d')

    def get_maps(self, score, micro_img, particle_windows, non_noise_windows):
        idx = np.argsort(-np.reshape(score, (np.prod(score.shape)), 'F'))
        
        y = idx % score.shape[0]
        x = np.floor(idx/score.shape[0])
        
        bw_mask_p = np.zeros((micro_img.shape[0], micro_img.shape[1]))
        
        begin_row_idx = y*int(self.query_size / 2)
        end_row_idx = np.minimum(y * int(self.query_size / 2) + self.query_size, bw_mask_p.shape[0] * np.ones(y.shape[0]))
        begin_col_idx = x*int(self.query_size / 2)
        end_col_idx = np.minimum(x * int(self.query_size / 2) + self.query_size, bw_mask_p.shape[1] * np.ones(x.shape[0]))
        
        begin_row_idx = begin_row_idx.astype(int)
        end_row_idx = end_row_idx.astype(int)
        begin_col_idx = begin_col_idx.astype(int)
        end_col_idx = end_col_idx.astype(int)
        
        for j in range(0, particle_windows.astype(int)):
            bw_mask_p[begin_row_idx[j]:end_row_idx[j], begin_col_idx[j]:end_col_idx[j]] = np.ones(
                end_row_idx[j] - begin_row_idx[j], end_col_idx[j] - begin_col_idx[j])

        bw_mask_n = np.copy(bw_mask_p)
        for j in range(particle_windows.astype(int), non_noise_windows.astype(int)):
            bw_mask_n[begin_row_idx[j]:end_row_idx[j], begin_col_idx[j]:end_col_idx[j]] = np.ones(
                end_row_idx[j] - begin_row_idx[j], end_col_idx[j] - begin_col_idx[j])
            
        return bw_mask_p, bw_mask_n
