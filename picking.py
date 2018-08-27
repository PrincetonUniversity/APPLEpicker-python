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
    particleSize = 0
    maxSize = 0
    minSize = 0
    querySize = 0
    tau1 = 0
    tau2 = 0
    MOA = 0
    containerSize = 0
    output_directory = ''
    filenames = ''

    def initialize_parameters(self, particleSize, maxSize, minSize, querySize, tau1, tau2, MOA, containerSize, filenames, output_directory):
        self.particleSize = int(particleSize/2)
        self.maxSize = int(maxSize/4)
        self.minSize = int(minSize/4)
        self.querySize = int(querySize/2)
        self.querySize = self.querySize - self.querySize%2
        self.tau1 = tau1
        self.tau2 = tau2
        self.MOA = int(MOA/2)
        self.containerSize = int(containerSize/2)
        self.filenames = filenames
        self.output_directory = output_directory
        
        self.querySize = self.querySize - self.querySize%2

    def read_mrc(self):
        mrc = mrcfile.open(self.filenames, mode='r+', permissive=True) 
        microImg = mrc.data 
        mrc.close()
        microImg = microImg.astype('float')

        microImg = microImg[99:-100, 99:-100]
        microImg = misc.imresize(microImg, 0.5, mode='F', interp='cubic')   

        gaussFilt = ApplePickerHelper.gaussian_filter(15, 0.5)
        microImg = signal.correlate(microImg, gaussFilt, 'same')       
        
        microImg = np.double(microImg)
        return microImg
    
    def query_score(self, microImg):
            
        queryBox = ApplePickerHelper.extract_query(microImg, int(self.querySize / 2))
        
        
        out_shape = (queryBox.shape[0], queryBox.shape[1], queryBox.shape[2], queryBox.shape[3] // 2 + 1)
        queryBoxA = np.empty(out_shape, dtype='complex128')
        fft_class_f = pyfftw.FFTW(queryBox, queryBoxA, axes=(2, 3), direction='FFTW_FORWARD')
        fft_class_f(queryBox, queryBoxA)                                       
        queryBox = np.conj(queryBoxA) 

        referenceBoxA = ApplePickerHelper.extract_references(microImg, self.querySize, self.containerSize)
        out_shape2 = (referenceBoxA.shape[0], referenceBoxA.shape[1], referenceBoxA.shape[-1] // 2 + 1)
        
        referenceBox = np.empty(out_shape2, dtype='complex128')
        fft_class_f2 = pyfftw.FFTW(referenceBoxA, referenceBox, axes=(1, 2), direction='FFTW_FORWARD')
        fft_class_f2(referenceBoxA, referenceBox)  
        
        convMap = np.zeros((referenceBox.shape[0], queryBox.shape[0], queryBox.shape[1]))
        
        window_t = np.empty(queryBox.shape, dtype=queryBox.dtype)
        cc = np.empty((queryBox.shape[0], queryBox.shape[1], queryBox.shape[2], 2*queryBox.shape[3]-2), dtype=microImg.dtype)
        fft_class = pyfftw.FFTW(window_t, cc, axes=(2, 3), direction='FFTW_BACKWARD')

        for index in range(0, referenceBox.shape[0]):
            np.multiply(referenceBox[index], queryBox, out=window_t)
            fft_class(window_t, cc)   
            convMap[index] = cc.real.max((2, 3)) - cc.real.mean((2,3))
        
        convMap = np.transpose(convMap, (1,2,0))

        minVal = np.amin(convMap)
        maxVal = np.amax(convMap)
        thresh = minVal + (maxVal - minVal)/20;
        
        h = convMap>=thresh
        score = np.sum(h, axis=2)

        return score

    def run_svm(self, microImg, score):
        particleWindows = np.floor(self.tau1)
        nonNoiseWindows = np.ceil(self.tau2)
        bwMask_p, bwMask_n = Picker.get_maps(self, score, microImg, particleWindows, nonNoiseWindows)
 
        x, y = ApplePickerHelper.get_training_set(microImg, bwMask_p, bwMask_n, self.querySize)

        scaler = preprocessing.StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)
        classify = svm.SVC(C=1, kernel='rbf', gamma=0.5, class_weight='balanced')
        classify.fit(x, y)                                                  # train SVM classifier
        
        meanAll, stdAll = ApplePickerHelper.moments(microImg, self.querySize)
                                                                    
        meanAll = meanAll[self.querySize-1:-(self.querySize-1), self.querySize-1:-(self.querySize-1)]
        stdAll = stdAll[self.querySize-1:-(self.querySize-1), self.querySize-1:-(self.querySize-1)]

        meanAll = np.reshape(meanAll, (np.prod(meanAll.shape), 1), 'F')
        stdAll = np.reshape(stdAll, (np.prod(stdAll.shape), 1), 'F')
        clsInput = np.concatenate((meanAll, stdAll), axis=1)
        clsInput = scaler.transform(clsInput)

        # compute classification for all possible windows in micrograph
        segmentation = classify.predict(clsInput)

        segmentation = np.reshape(segmentation, (int(np.sqrt(segmentation.shape[0])), int(np.sqrt(segmentation.shape[0]))))
        segmentation = segmentation.copy()
        return segmentation
    
    def morphology_ops(self, segmentation):
        if (ndimage.morphology.binary_fill_holes(segmentation)==np.ones(segmentation.shape)).all():
            segmentation[0:100, 0:100] = np.zeros((100,100))
            
        segmentation = ndimage.morphology.binary_fill_holes(segmentation)
        y,x = np.ogrid[-self.minSize:self.minSize+1, -self.minSize:self.minSize+1]
        element = x*x+y*y <= self.minSize*self.minSize
        segmentation_e = ndimage.morphology.binary_erosion(segmentation, element) 
        
        y,x = np.ogrid[-self.maxSize:self.maxSize+1, -self.maxSize:self.maxSize+1]
        element = x*x+y*y <= self.maxSize*self.maxSize
        segmentation_o = ndimage.morphology.binary_erosion(segmentation, element)
        segmentation_o = np.reshape(segmentation_o, (segmentation_o.shape[0], segmentation_o.shape[1], 1)) 
        
        sizeConst, num_features = ndimage.label(segmentation_e, np.ones((3,3)))
        sizeConst = np.reshape(sizeConst, (sizeConst.shape[0], sizeConst.shape[1], 1))
        labels = np.unique(sizeConst*segmentation_o)
        idx = np.where(labels!=0)
        labels = np.take(labels, idx)
        labels = np.reshape(labels, (1,1,np.prod(labels.shape)))
        
        matrix1 = np.repeat(sizeConst, labels.shape[2], 2)
        matrix2 = np.repeat(labels, matrix1.shape[0], 0)
        matrix2 = np.repeat(matrix2, matrix1.shape[1], 1)
        
        matrix3 = np.equal(matrix1, matrix2)
        matrix4 = np.sum(matrix3, 2)
        
        segmentation_e[np.where(matrix4==1)] = 0
        
        return segmentation_e
    
    def extract_particles(self, segmentation):
        segmentation = segmentation[self.querySize//2-1:-self.querySize//2, self.querySize//2-1:-self.querySize//2]
        labeledSegments, num_features = ndimage.label(segmentation, np.ones((3,3)))
        values, repeats = np.unique(labeledSegments, return_counts=True)
        
        vals2remove = np.where(repeats>((self.querySize)**2))
        values = np.take(values, vals2remove)
        values = np.reshape(values, (1,1,np.prod(values.shape)))
        
        labeledSegments = np.reshape(labeledSegments, (labeledSegments.shape[0], labeledSegments.shape[1], 1))
        matrix1 = np.repeat(labeledSegments, values.shape[2], 2)
        matrix2 = np.repeat(values, matrix1.shape[0], 0)
        matrix2 = np.repeat(matrix2, matrix1.shape[1], 1)
        
        matrix3 = np.equal(matrix1, matrix2)
        matrix4 = np.sum(matrix3, 2)
        
        segmentation[np.where(matrix4==1)] = 0
        labeledSegments, num_features = ndimage.label(segmentation, np.ones((3,3)))
        
        maxVal = np.amax(np.reshape(labeledSegments, (np.prod(labeledSegments.shape))))
        center = ndimage.measurements.center_of_mass(segmentation, labeledSegments, np.arange(1, maxVal))
        center = np.rint(center)
        
        img = np.zeros((segmentation.shape[0], segmentation.shape[1]))
        img[center[:,0].astype(int), center[:,1].astype(int)] = 1
        y,x = np.ogrid[-self.MOA:self.MOA+1, -self.MOA:self.MOA+1]
        element = x*x+y*y <= self.MOA*self.MOA
        img = ndimage.morphology.binary_dilation(img, structure=element)
        labeledImg, num_features = ndimage.label(img, np.ones((3,3)))
        values, repeats = np.unique(labeledImg, return_counts=True)
        y = np.where(repeats==np.count_nonzero(element))
        y = np.array(y)
        y = y.astype(int)
        y = np.reshape(y, (np.prod(y.shape)))
        y = y - 1
        center = center[y, :]
        
        center = center + (self.querySize//2 - 1)*np.ones(center.shape)
        center = center + (self.querySize//2 - 1)*np.ones(center.shape)
        center = center + np.ones(center.shape)
        center = 2*center
        center = center + 99*np.ones(center.shape)
        
        nameList = self.filenames.split("/")
        name = nameList[-1].split(".")
        nameStr = name[0]
        
        f = open(self.output_directory + '/' + nameStr +"_applepick.star", "w+")
        np.savetxt(f, ["data_root\n\nloop_\n_rlnCoordinateY #1\n_rlnCoordinateX #2"], fmt='%s')
        np.savetxt(f, center, fmt='%d %d')
        f.close()

    def get_maps(self, score, microImg, particleWindows, nonNoiseWindows):
        idx = np.argsort(-np.reshape(score, (np.prod(score.shape)), 'F'))
        
        y = idx % score.shape[0]
        x = np.floor(idx/score.shape[0])
        
        bwMask_p    = np.zeros((microImg.shape[0], microImg.shape[1]))
        
        beginRowIdx = y*int(self.querySize/2)
        endRowIdx   = np.minimum(y*int(self.querySize/2)+self.querySize, bwMask_p.shape[0]*np.ones(y.shape[0]))
        beginColIdx = x*int(self.querySize/2)
        endColIdx   = np.minimum(x*int(self.querySize/2)+self.querySize, bwMask_p.shape[1]*np.ones(x.shape[0]))
        
        beginRowIdx = beginRowIdx.astype(int)
        endRowIdx   = endRowIdx.astype(int)
        beginColIdx = beginColIdx.astype(int)
        endColIdx   = endColIdx.astype(int)
        
        for j in range(0, particleWindows.astype(int)):
            bwMask_p[beginRowIdx[j]:endRowIdx[j], beginColIdx[j]:endColIdx[j]] = np.ones(endRowIdx[j]-beginRowIdx[j], endColIdx[j]-beginColIdx[j])

        bwMask_n = np.copy(bwMask_p)
        for j in range(particleWindows.astype(int), nonNoiseWindows.astype(int)):
            bwMask_n[beginRowIdx[j]:endRowIdx[j], beginColIdx[j]:endColIdx[j]] = np.ones(endRowIdx[j]-beginRowIdx[j], endColIdx[j]-beginColIdx[j])
            
        return bwMask_p, bwMask_n
