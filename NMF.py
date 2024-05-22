#!/usr/bin/env python
# coding: utf-8

# # From <Automatic Microscopic Image Analysis by Moving Window Local Fourier Transform and Machine Learning>
# 
# ## thank Benedykt R. Jany
# 
# 
# # Institute of Physics Jagiellonian University in Krakow, Poland
# 
# ## To run this code first you have to install HyperSpy https://hyperspy.org/
# 

import matplotlib

import numpy as np
from numpy.lib.stride_tricks import as_strided
import imageio # supported file types https://imageio.readthedocs.io/en/stable/formats.html
import os
matplotlib.use('pdf')
import hyperspy.api as hs
from scipy.signal import argrelextrema
import pdb
import cv2

hs.preferences.GUIs.warn_if_guis_are_missing = False
hs.preferences.save()

def NMF(img, elementsize, step):
    # 将OpenCV的NumPy数组转换为imageio格式
    imdata = imageio.core.util.Array(img)
    # imdata = imageio.core.asarray(img)

    im = hs.signals.Signal2D(imdata)

    # resize
    imshape = im.axes_manager.signal_shape
    wscale = imshape[0]/2048.
    im = im.rebin(scale=[wscale, wscale])
    imdata = im.data

    im.plot(scalebar=False, axes_off=True)


    # elementsize = 128 #128 default for ~2000x2000 pixels image

    # shape of the elements on which you want to perform the operation (e.g. Fourier Transform)
    ws = np.arange(elementsize*elementsize).reshape(elementsize, elementsize) 

    imdataW = as_strided(imdata, shape=(int((imdata.shape[0]-ws.shape[0]+1)/step),int((imdata.shape[1]-ws.shape[1]+1)/step),ws.shape[0],ws.shape[1]), strides=(imdata.strides[0]*step,imdata.strides[1]*step,imdata.strides[0],imdata.strides[1]))

    hanningf = np.hanning(elementsize)
    hanningWindow2d = np.sqrt(np.outer(hanningf, hanningf))


    imdataWfft = np.fft.fftshift(np.abs(np.fft.fft2(hanningWindow2d*imdataW))**2, axes=(2,3))

    imdataWfft = imdataWfft+10000 #adding offset to prevent 0

    imdataWfft = np.log(np.abs(imdataWfft))

    imWindowFFT = hs.signals.Signal2D(imdataWfft)

    imWindowFFT.decomposition(algorithm="sklearn_pca")

    screedata = imWindowFFT.get_explained_variance_ratio().data

    grad = np.gradient(screedata)

    #局部最大值的坐标
    gradLocalMaxima0 = argrelextrema(grad, np.greater)
    gradLocalMaxima = [x + 1 for x in gradLocalMaxima0[0]] # add 1 due to the array indexing from 0


    NComponents = gradLocalMaxima[0]

    NComponents = imWindowFFT.estimate_elbow_position()

    imWindowFFT.decomposition(algorithm="nmf", output_dimension=NComponents)


    loadingsS = imWindowFFT.get_decomposition_loadings()
    factorsS =  imWindowFFT.get_decomposition_factors()
    
    loadings = []
    factors = []
    for i in range(loadingsS.data.shape[0]):
        imgloading = (loadingsS.data[i]-loadingsS.data[i].min())/(loadingsS.data[i].max()-loadingsS.data[i].min())
        imgfactor = (factorsS.data[i]-factorsS.data[i].min())/(factorsS.data[i].max()-factorsS.data[i].min())
        loadings.append(imgloading)
        
        start_index = (elementsize - 64) // 2
        center_crop = imgfactor[start_index:start_index+64, start_index:start_index+64]
        factors.append(center_crop*255)
    return loadings, factors


