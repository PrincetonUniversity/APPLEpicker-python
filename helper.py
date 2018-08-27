# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 11:25:48 2018

@author: Ayelet Heimowitz, Itay Sason
"""

import numpy as np
import pyfftw


class ApplePickerHelper:
    
    def gaussian_filter(sizeFilter, std):

        y,x = np.mgrid[-(sizeFilter-1)//2:(sizeFilter-1)//2+1,-(sizeFilter-1)//2:(sizeFilter-1)//2+1]

        response = np.exp(-np.square(x) - np.square(y) / (2*(std**2)))/(np.sqrt(2*np.pi)*std)
        response[response < np.finfo('float').eps] = 0
        
        response = response/response.sum() # Normalize so sum is 1
            
        return response
    
    def extract_windows(img, blockSize):
        
        # get size of image
        Sx = img.shape[1]
        Sy = img.shape[0]
        
        blockElements = blockSize**2
        
        # keep only the portion of the image that can be slpit into blocks with no remainder
        truncX = Sx%blockSize
        truncY = Sy%blockSize
        
        img = img[:Sy-truncY, :Sx-truncX]
        dim3_size = np.sqrt(np.prod(img.shape)//(blockSize**2))
        
        img = np.reshape(img, (blockSize,dim3_size.astype(int),blockSize, dim3_size.astype(int)), 'F')
        img = np.transpose(img, (0,2,1,3))
        img = np.reshape(img, (img.shape[0]*img.shape[1], img.shape[2], img.shape[3]), 'F')
        img = np.reshape(img, (img.shape[0], img.shape[1]*img.shape[2]), 'F')
        img = img.copy()
            
        return img
        
    def extract_query(img, blockSize):
        
        Sx = img.shape[1]
        Sy = img.shape[0]
        
        # keep only the portion of the image that can be slpit into blocks with no remainder
        truncX = Sx%blockSize
        truncY = Sy%blockSize
 
        blocks = img[:Sy-truncY, :Sx-truncX]

        dim3_size = np.sqrt(np.prod(blocks.shape)//(blockSize**2))
        blocks = np.reshape(blocks, (blockSize, dim3_size.astype(int) ,blockSize, dim3_size.astype(int)), 'F')

        blocks = np.transpose(blocks, (0,2,1,3))

        blocks = np.reshape(blocks, (blocks.shape[0], blocks.shape[1], -1), 'F')

        blocks = np.concatenate((blocks, np.concatenate((blocks[:,:,1:], np.reshape(blocks[:,:,0], (blocks.shape[0], blocks.shape[1], 1))), axis=2)), axis=0)        

        temp = np.concatenate((blocks[:,:,int(np.floor(2*img.shape[1]/2/blockSize)):], blocks[:,:,0:int(np.floor(2*img.shape[1]/2/blockSize))]), axis=2)
        blocks = np.concatenate((blocks, temp), axis=1)
        blocks = np.reshape(blocks, (2*blockSize, 2*blockSize, int(np.floor(2*img.shape[0]/2/blockSize)), int(np.floor(2*img.shape[1]/2/blockSize))), 'F')
        blocks = blocks[:, :, 0:blocks.shape[2]-1, 0:blocks.shape[3]-1]
        
        blocks = np.transpose(blocks, (2,3,0,1))
        blocks = blocks.copy()
        return blocks
    
    def extract_references(img, querySize, containerSize):
        
        numContainersRow = int(np.floor(img.shape[0]/containerSize))
        numContainersCol = int(np.floor(img.shape[1]/containerSize))
        
        windows = np.zeros((numContainersRow*numContainersCol*4, querySize, querySize))
        winIdx = 0
        
        meanAll, stdAll = ApplePickerHelper.moments(img, querySize)
        
        for yContain in range(1, numContainersRow+1):
            for xContain in range(1, numContainersCol+1):
                
                temp = img[(yContain-1)*containerSize : min(img.shape[0], yContain*containerSize), 
                           (xContain-1)*containerSize : min(img.shape[1], xContain*containerSize)]
        
                meanContain = meanAll[(yContain-1)*containerSize+querySize-1:min(meanAll.shape[0]-querySize, (yContain-1)*containerSize+containerSize),
                                      (xContain-1)*containerSize+querySize-1:min(meanAll.shape[1]-querySize, (xContain-1)*containerSize+containerSize)]
                
                
                stdContain = stdAll[(yContain-1)*containerSize+querySize-1:min(meanAll.shape[0]-querySize, (yContain-1)*containerSize+containerSize),
                                      (xContain-1)*containerSize+querySize-1:min(meanAll.shape[1]-querySize, (xContain-1)*containerSize+containerSize)]
                
                y,x = np.where(meanContain==meanContain.max())
                windows[winIdx,:,:] = temp[int(y):int(y+querySize), int(x):int(x+querySize)]
                
                winIdx = winIdx + 1
                y,x = np.where(meanContain==meanContain.min())
                windows[winIdx,:,:] = temp[int(y):int(y+querySize), int(x):int(x+querySize)]
                
                winIdx = winIdx + 1
                y,x = np.where(stdContain==stdContain.max())
                windows[winIdx,:,:] = temp[int(y):int(y+querySize), int(x):int(x+querySize)]
                
                winIdx = winIdx + 1
                y,x = np.where(stdContain==stdContain.min())
                windows[winIdx,:,:] = temp[int(y):int(y+querySize), int(x):int(x+querySize)]
                
                winIdx = winIdx + 1
            
        windows = windows.copy()
        return windows
    
    def get_training_set(microImg, bwMask_p, bwMask_n, N):
        
        nonOverlap = ApplePickerHelper.extract_windows(microImg, N)
        
        windows = nonOverlap.copy()
        indicate = ApplePickerHelper.extract_windows(bwMask_p, N)
        r, c = np.where(indicate==0)
        c = np.setdiff1d(np.arange(0, indicate.shape[1]), c)
        windows = windows.take(c, 1)
        p_mu = np.mean(windows, axis=0)
        p_std = np.std(windows, axis=0)
        
        windows = nonOverlap.copy()
        indicate = ApplePickerHelper.extract_windows(bwMask_n, N)
        r, c = np.where(indicate==1)
        c = np.setdiff1d(np.arange(0, indicate.shape[1]), c)
        windows = windows.take(c, 1)
        n_mu = np.mean(windows, axis=0)
        n_std = np.std(windows, axis=0)
        
        p_mu = np.reshape(p_mu, (p_mu.shape[0], 1))
        p_std = np.reshape(p_std, (p_std.shape[0], 1))
        n_mu = np.reshape(n_mu, (n_mu.shape[0], 1))
        n_std = np.reshape(n_std, (n_std.shape[0], 1))
        
        x = np.concatenate((p_mu, p_std), axis=1)
        x = np.concatenate((x, np.concatenate((n_mu, n_std), axis=1)), axis=0)
        
        y = np.concatenate((np.ones(p_mu.shape[0]), np.zeros(n_mu.shape[0])), axis=0)
        
        return x,y
    
    def moments(img, querySize):

        filt = np.ones((querySize, querySize)) / (querySize * querySize)
        filt = np.pad(filt, (0,img.shape[0]-1), 'constant', constant_values=(0, 0))
        padImg = np.pad(img, (0,querySize-1), 'constant', constant_values=(0, 0))
        padImg_square = np.square(padImg)
        filtFreq = np.empty(padImg.shape, dtype = 'complex128')
        imgFreq = np.empty(padImg.shape, dtype = 'complex128')
        
        fft_class = pyfftw.FFTW(filt.astype('complex128'), filtFreq, axes=(0, 1), direction='FFTW_FORWARD')

        fft_class(filt, filtFreq)
        fft_class(padImg, imgFreq)

        meanFreq = np.empty(filtFreq.shape, dtype=filtFreq.dtype)
        np.multiply(imgFreq, filtFreq, out=meanFreq)
        
        meanAll = np.empty(meanFreq.shape, dtype = meanFreq.dtype)
        fft_class2 = pyfftw.FFTW(meanFreq, meanAll, axes=(0, 1), direction='FFTW_BACKWARD')
        fft_class2(meanFreq, meanAll)
        meanAll = np.real(meanAll)

        imgVarFreq = np.empty(padImg_square.shape, dtype = 'complex128')
        varFreq = np.empty(padImg_square.shape, dtype = 'complex128')
        fft_class(padImg_square, imgVarFreq)

        np.multiply(imgVarFreq, filtFreq, out=varFreq)

        varAll = np.empty(varFreq.shape, dtype=varFreq.dtype)
        fft_class2(varFreq, varAll)
        varAll = np.real(varAll) - np.power(meanAll, 2)
        
        stdAll = np.sqrt(varAll)
        
        return meanAll, stdAll
