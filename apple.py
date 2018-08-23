#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 09:40:13 2018

@author: Ayelet Heimowitz, Itay Sason
"""


from tkinter.ttk import *
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter import simpledialog

import numpy as np
import scipy as sci

import svmpy
svmpy.svmpy = svmpy
from picking import picking

import time
import os

from multiprocessing import Pool, Process
from functools import partial

#1- pyCUDA
#2- numba and NumbaPro
#3- Theano 

class apple(ttk.Frame):
    pSize = 0
    qSize = 0
    cSize = 450
    maxSize = 0
    minSize = 0
    tau1 = 0
    tau2 = 0
    moa = 0
    directory_in=''
    directory_out=''
    proc = 1
    
    ###############################################################################################
    # __init__ initializes the GUI                                                                #
    ###############################################################################################

    def __init__(self, master):

        ttk.Frame.__init__(self,
                          master)
        # Set the title
        self.master.title('APPLE picker')
        
        self.pack_propagate(0)
        self.pack(side=TOP)
        
        self.columnconfigure(0, minsize=150)
        self.columnconfigure(1, minsize=150)
        self.columnconfigure(2, minsize=150)
        
        # create labels in GUI
        self.l0 = Label(self, text="Particle Size (pixels)").grid(row=0, column=0)
        
        # create a box for the user to supply particle size in GUI
        self.e1=Entry(self, width=6)
        self.e1.grid(row=0, column=1)
        self.e1.config(justify=CENTER)
        self.e1.pack()
        
        # create button in GUI. Function: calculate default parameter values
        self.go_button = ttk.Button(self, text='Go',command=self.fill_data)
        
        self.go_button.pack()
        self.go_button.grid(row=0, column=2)
        
        # create labels for APPLE picking parameters
        Label(self, text="Query Image Size").grid(row=2, column=0)
        Label(self, text="Maximum Particle Size").grid(row=3, column=0)
        Label(self, text="Minimum Particle Size").grid(row=4, column=0)
        Label(self, text=u'\u03C4\u2081').grid(row=5, column=0)
        Label(self, text=u'\u03C4\u2082').grid(row=6, column=0)
        Label(self, text="Minimal Overlap Allowed").grid(row=7, column=0)
        Label(self, text="Container Size").grid(row=8, column=0)
        Label(self, text="Input Directory").grid(row=9, column=0)
        Label(self, text="Output Directory").grid(row=10, column=0)
        Label(self, text="Number of Processors").grid(row=11, column=0)
        
        # create button in GUI. Function: begin particle picking
        self.apple_button = ttk.Button(self,
                                   text='Pick Particles',
                                   command=self.pickParticles)
        
        
        # Put the controls on the form
        self.apple_button.pack()
        self.apple_button.grid(row=13, column=1)
        
    ###############################################################################################
    # fill_data outputs default values for the APPLE picking parameters                           #
    ###############################################################################################        
    def fill_data(self):
        
        # get the particle size
        value = self.e1.get()
        try:
            value = int(value)
        except ValueError:
            messagebox.showerror("Error", "Input must be an integer")
            return
        
        self.setButtons()
            
        self.pSize = value
        
        # stringVars to use in the GUI
        self.qVar = StringVar()
        self.mxVar = StringVar()
        self.mnVar = StringVar()
        self.tau1Var = StringVar()
        self.tau2Var = StringVar()
        self.moaVar = StringVar()
        self.cSizeVar = StringVar()
        self.dir_name_in = StringVar()
        self.dir_name_out = StringVar()
        self.poolSize = StringVar()
        
        # query window size
        value = float(self.e1.get())
        value = value * 2 / 3
        value = np.round(value)
        value = value - value%4
        value = int(value)
        
        self.qVar.set(value)
        self.l1 = Label(self, textvariable=self.qVar).grid(row=2, column=1)
        self.qSize = value
        
        # max particle size
        value = int(self.e1.get())
        value = int(4*value)
        self.mxVar.set(value)
        self.l2 = Label(self, textvariable=self.mxVar).grid(row=3, column=1)
        self.maxSize = value
        
        # minimal particle size
        value = self.qSize
        self.mnVar.set(int(value/4))
        self.l3 = Label(self, textvariable=self.mnVar).grid(row=4, column=1)
        self.minSize = value
        
        # maximum overlap between 2 particles
        value = float(self.e1.get())
        value = value/10
        value = int(value)
        self.moaVar.set(value)
        self.l4 = Label(self, textvariable=self.moaVar).grid(row=7, column=1)
        self.moa = value
        
        # tau_1 size
        qBox = (4000**2)/(self.qSize**2)*4
        value = int(qBox*3/100)
        self.tau1Var.set(value)
        self.l5 = Label(self, textvariable=self.tau1Var).grid(row=5, column=1)
        self.tau1 = value
        
        # tau_2 size
        value =  int(qBox*30/100)
        self.tau2Var.set(value)
        self.l6 = Label(self, textvariable=self.tau2Var).grid(row=6, column=1)
        self.tau2 = value
        
        self.cSizeVar.set(self.cSize)
        self.l7 = Label(self, textvariable=self.cSizeVar).grid(row=8, column=1)
        
        self.dir_name_in.set(self.directory_in)
        self.l8 = Label(self, textvariable=self.dir_name_in).grid(row=9, column=1)
        
        self.dir_name_out.set(self.directory_out)
        self.l9 = Label(self, textvariable=self.dir_name_out).grid(row=10, column=1)
        
        self.poolSize.set(self.proc)
        self.l10 = Label(self, textvariable=self.poolSize).grid(row=11, column=1)
        
    ###############################################################################################
    # setButtons initializes the set buttons in the GUI                                           #
    ###############################################################################################
    def setButtons(self):
    
        self.set_button_qsize = ttk.Button(self,
                                   text='Set',
                                   command=self.getQsize)
        
        self.set_button_qsize.pack()
        self.set_button_qsize.grid(row=2, column=2)
        
        self.set_button_maxSize = ttk.Button(self,
                                   text='Set',
                                   command=self.getMXsize)
        
        self.set_button_maxSize.pack()
        self.set_button_maxSize.grid(row=3, column=2)
        
        self.set_button_minSize= ttk.Button(self,
                                   text='Set',
                                   command=self.getMNsize)
        
        self.set_button_minSize.pack()
        self.set_button_minSize.grid(row=4, column=2)
        
        self.set_button_tau1= ttk.Button(self,
                                   text='Set',
                                   command=self.getTau1size)
        
        self.set_button_tau1.pack()
        self.set_button_tau1.grid(row=5, column=2)
        
        self.set_button_tau2= ttk.Button(self,
                                   text='Set',
                                   command=self.getTau2size)
        
        self.set_button_tau2.pack()
        self.set_button_tau2.grid(row=6, column=2)
        
        self.set_button_moa= ttk.Button(self,
                                   text='Set',
                                   command=self.getMOAsize)
        
        self.set_button_moa.pack()
        self.set_button_moa.grid(row=7, column=2)
        
        self.set_button_container= ttk.Button(self,
                                   text='Set',
                                   command=self.getCsize)
        
        self.set_button_container.pack()
        self.set_button_container.grid(row=8, column=2)
        
        self.set_button_directory= ttk.Button(self,
                                   text='Set',
                                   command=self.getPath)
        
        self.set_button_directory.pack()
        self.set_button_directory.grid(row=9, column=2)
        
        self.set_button_directory_out= ttk.Button(self,
                                   text='Set',
                                   command=self.getPath_out)
        
        self.set_button_directory_out.pack()
        self.set_button_directory_out.grid(row=10, column=2)
        
        self.set_button_pool= ttk.Button(self,
                                   text='Set',
                                   command=self.setPool)
        
        self.set_button_pool.pack()
        self.set_button_pool.grid(row=11, column=2)
        
    ###############################################################################################
    # getQsize allows to change the size of query images                                          #
    ###############################################################################################

    def getQsize(self):
        
        value = simpledialog.askstring("Query Size", "Please input new size",
                                parent=self)
        try:
            value = int(value)
        except ValueError:
            messagebox.showerror("Error", "Input must be an integer")
            return
        
        self.qSize = value
        self.qVar.set(value)

    
    ###############################################################################################
    # getMXsize allows to change the maximum allowed particle size                                #
    ###############################################################################################
    def getMXsize(self):
        
        value = simpledialog.askstring("Max Particle Size", "Please input new size",
                                parent=self)
        try:
            value = int(value)
        except ValueError:
            messagebox.showerror("Error", "Input must be an integer")
            return
        
        self.maxSize = value
        self.mxVar.set(value)


    ###############################################################################################
    # getMNsize allows to change the minumum allowed particle size                                #
    ###############################################################################################    
    def getMNsize(self):
        
        value = simpledialog.askstring("Min Particle Size", "Please input new size",
                                parent=self)
        
        try:
            value = int(value)
        except ValueError:
            messagebox.showerror("Error", "Input must be an integer")
            return
        
        self.minSize = value
        self.mnVar.set(value)
    
    ###############################################################################################
    # getTau1size allows to control the particle training examples                                #
    ###############################################################################################
    def getTau1size(self):
        
        value = simpledialog.askstring("\u03C4\u2081", "Please input \u03C4\u2081",
                                parent=self)
        
        try:
            value = int(value)
        except ValueError:
            messagebox.showerror("Error", "Input must be an integer")
            return
        
        self.tau1 = value
        self.tau1Var.set(value)
        
    
    ###############################################################################################
    # getTau2size allows to control the noise training examples                                   #
    ###############################################################################################    
    def getTau2size(self):
        
        value = simpledialog.askstring("\u03C4\u2082", "Please input \u03C4\u2082",
                                parent=self)
        
        try:
            value = int(value)
        except ValueError:
            messagebox.showerror("Error", "Input must be an integer")
            return
        
        self.tau2 = value
        self.tau2Var.set(value)
 
    ###############################################################################################
    # getMOAsize allows to change the maximum allowed overlap between particles                   #
    ###############################################################################################
    def getMOAsize(self):
        
        value = simpledialog.askstring("\u03C4\u2081", "Please input \u03C4\u2081",
                                parent=self)
        
        try:
            value = int(value)
        except ValueError:
            messagebox.showerror("Error", "Input must be an integer")
            return
        
        self.moa = value
        self.moaVar.set(value)
        
    ###############################################################################################
    # getCsize allows to change the container size                                                #
    ###############################################################################################
    def getCsize(self):
        
        value = simpledialog.askstring("Container Size", "Please input new size",
                                parent=self)
        
        try:
            value = int(value)
        except ValueError:
            messagebox.showerror("Error", "Input must be an integer")
            return
        
        self.cSize = value
        self.cSizeVar.set(value)
        
    ###############################################################################################
    # getPath allows to get path to directory of micrpgraphs                                      #
    ###############################################################################################
    def getPath(self):
        directory = filedialog.askdirectory()
        self.directory = directory
        self.dir_name_in.set(directory)
        return
    
    ##############################################################################################
    # getPath allows to get path to directory of micrpgraphs                                      #
    ###############################################################################################
    def getPath_out(self):
        directory = filedialog.askdirectory()
        self.directory_out = directory
        self.dir_name_out.set(directory)
        return
    
    def setPool(self):
        value = simpledialog.askstring("Container Size", "Please input new size",
                                parent=self)
        
        try:
            value = int(value)
        except ValueError:
            messagebox.showerror("Error", "Input must be an integer")
            return
        
        self.proc = value
        self.poolSize.set(value)
            
    def checkParameters(self):

        if self.maxSize<1:
            messagebox.showerror("Error", "Max particle size must be a positive integer.")
            return 1
        if self.maxSize>3000:
            messagebox.showerror("Error", "Max particle size is too large.")
            return 1
        
        if self.qSize<1:
            messagebox.showerror("Error", "Query image size must be a positive integer.")
            return 1
        if self.qSize>3000:
            messagebox.showerror("Error", "Query image size is too large.")
            return 1
        
        if self.pSize<5:
            messagebox.showerror("Error", "Particle size too small.")
            return 1
        if self.pSize>3000:
            messagebox.showerror("Error", "Particle size too large.")
            return 1
        
        if self.minSize<1:
            messagebox.showerror("Error", "Min particle size must be a positive integer.")
            return 1
        if self.minSize>3000:
            messagebox.showerror("Error", "Min particle size is too large.")
            return 1
        
        if self.tau1<0:
            messagebox.showerror("Error", "\u03C4\u2081 must be a positive integer.")
            return 1
        if self.tau1>(4000/self.qSize*2)**2:
            messagebox.showerror("Error", "\u03C4\u2081 is too large.")
            return 1
        
        if self.tau2<0:
            messagebox.showerror("Error", "\u03C4\u2082 must be a positive integer.")
            return 1
        if self.tau2>(4000/self.qSize*2)**2:
            messagebox.showerror("Error", "\u03C4\u2082 is too large.")
            return 1
        
        if self.moa<1:
            messagebox.showerror("Error", "Min overlap must be a positive integer.")
            return 1
        if self.moa>3000:
            messagebox.showerror("Error", "Min overlap is too large.")
            return 1
        
        if self.cSize*2+200>4000:
            messagebox.showerror("Error", "Container size is too big.")
            return 1
        if self.cSize<self.pSize:
            messagebox.showerror("Error", "Container size must exceed particle size.")
            return 1
        if self.pSize<self.qSize:
            messagebox.showerror("Error", "Particle size must exceed query image size.")
            return 1
        if self.dir_name_in=='':
            messagebox.showerror("Error", "No input directory selected.")
            return 1
        if self.dir_name_out=='':
            messagebox.showerror("Error", "No output directory selected.")
            return 1
        if self.proc<1:
            messagebox.showerror("Error", "Please select at least one processor.")
            return 1
        return 0

    ###############################################################################################
    # pickParticles: main functionality                                                           #
    ###############################################################################################
    def pickParticles(self):

        # check all inputs
        if self.checkParameters() == 1:
            return

        # get the names of all fines in input directory
        filenames = os.listdir(self.directory)

        data = list()
        data.append(self.directory)
        data.append(self.pSize)
        data.append(self.maxSize)
        data.append(self.minSize)
        data.append(self.qSize)
        data.append(self.tau1)
        data.append(self.tau2)
        data.append(self.moa)
        data.append(self.cSize)
        data.append(self.directory_out)

        pool = Pool(processes=self.proc)
        partial_func=partial(apple.process_micrograph, data)
        pool.map(partial_func, filenames)
        pool.terminate()

        # for now only
        #        apple.process_micrograph(data, filenames[1])



        # reset parameters in GUI
        messagebox.showinfo("APPLE picker", "particle picking complete")
        self.qVar.set('')
        self.mxVar.set('')
        self.mnVar.set('')
        self.tau1Var.set('')
        self.tau2Var.set('')
        self.moaVar.set('')
        self.cSizeVar.set('')
        self.dir_name_in.set('')
        self.dir_name_out.set('')
        self.directory_out = ''
        self.directory = ''
        self.cSize = -1
        self.moa = -1
        self.tau1 = -1
        self.tau2 = -1
        self.maxSize = -1
        self.minSize = -1
        self.pSize = -1
        self.qSize = -1

        return
        
    def process_micrograph(data, filenames):
        file_basename, file_extension = os.path.splitext(filenames)

        # parse filename and verify extension is ".mrc"
        if file_extension=='.mrc':  # todo use negative condition, use other func

                        
            myPicker = picking()
            
            directory     = data[0]
            pSize         = data[1]
            maxSize       = data[2]
            minSize       = data[3]
            qSize         = data[4]
            tau1          = data[5]
            tau2          = data[6]
            moa           = data[7]
            cSize         = data[8]
            directory_out = data[9]
            
            # add path to filename
            filename = directory + '/' + filenames
            
            # Initialize parameters for the APPLE picker
            picking.initializeParameters(myPicker, pSize, maxSize, minSize, qSize, tau1, tau2, moa, cSize, filename, directory_out)
            
            # update user
            print('Processing ', end='')
            nameParse = filenames.split("/")
            print(nameParse[-1])            

            # return .mrc file as a float64 array
            microImg = picking.readMRC(myPicker) # return a micrograph as an numpy array
        
            # bin and filter micrograph
#            microImg = picking.initialManipulations(myPicker, microImg) # filtering, binning, etc.
            
            # compute score for query images
            score = picking.queryScore(myPicker, microImg) # compute score using normalized cross-correlations
            flag = True
            
            while (flag):
                # train SVM classifier and classify all windows in micrograph
                segmentation = picking.runSVM(myPicker, microImg, score)
                
                # If all windows are classified identically, update tau_1 or tau_2
                if np.array_equal(segmentation, np.ones((segmentation.shape[0], segmentation.shape[1]))):
                    tau2 = tau2+1
                elif np.array_equal(segmentation, np.zeros((segmentation.shape[0], segmentation.shape[1]))):
                    tau1 = tau1+1
                else:
                    flag = False

            # discard suspected artifacts
            segmentation = picking.morphologyOps(myPicker, segmentation)
            
            # create output star file
            picking.extractParticles(myPicker, segmentation)
            
            return
        

    def run(self):
        ''' Run the app '''
        self.mainloop()
 
app = apple(Tk())
app.run()
