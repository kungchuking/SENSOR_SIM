import cv2
import numpy as np
from PIL import Image  
import time
import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import io

def solvePhase(image, matrix):
    
    H,W,N = image.shape
    newImage = image.reshape(H*W, N)
    wrappedPhase = np.linalg.lstsq(matrix.T,newImage.T, rcond=None)[0]
    
    
    cosVal = wrappedPhase[0,:] / np.sqrt(np.power(wrappedPhase[0,:],2) + np.power(wrappedPhase[1,:],2))
    
    albedo = np.sqrt(np.power(wrappedPhase[0,:],2) + np.power(wrappedPhase[1,:],2))
    unsignedPhase = np.arccos(cosVal)
    
    signedPhase = np.sign(wrappedPhase[1,:]) * unsignedPhase
    
    signedPhase= signedPhase.reshape(H,W)
    albedo = albedo.reshape(H,W)
    
    return signedPhase, albedo

def cleanupImage(bucket1,bucket2):
    
    return newbucket1, newbucket2

def demosaicFunction(image):
    return cv2.cvtColor(image, cv2.COLOR_BAYER_BG2BGR)

def demosaicStep(demosaicFunction, image):
    return demosaicFunction(image)

def demultiplexStep(bucket1,bucket2, multiplexIm):
    H,W, K = bucket1.shape
    false,N = multiplexIm.shape
    
    allMult = np.concatenate((multiplexIm, 1- multiplexIm), axis =0)
    allIms = np.concatenate((bucket1.reshape(H*W, K),bucket2.reshape(H*W, K)), axis = 1) 
    
    sol = np.linalg.lstsq(allMult, allIms.T, rcond=None)
    
    demulIms = sol[0].T
    return demulIms.reshape((H,W,N))
    
def refinePhaseWithDepthBounds(oldPhase, freq, lbPhase, ubPhase):    
    ub = freq*ubPhase - oldPhase;
    lb = freq*lbPhase - oldPhase;
    
    k_ub = ub/(2*np.pi);
    k_lb = lb/(2*np.pi);
    
    k_ub_new = np.floor(k_ub);
    k_lb_new = np.ceil(k_lb);
    
    choice = np.floor((k_ub_new - k_lb_new))
    
    k = choice + k_lb_new;
    phase = (oldPhase + k * 2*np.pi)/freq;
    
    return phase

def disparityFunction(corres, pos):
    return corres - 2.7*pos;
    
    
def computeDisparity(dispFunc, phase, projWidth):
    H,W = phase.shape;
    xVals = list(range(W))
    yVals = list(range(H))
    y, x = np.meshgrid(yVals, xVals, sparse=False, indexing='ij')
    newDisparity = phase * projWidth / (2*np.pi);
    disparity = dispFunc(newDisparity, y+1)
    
    return disparity
    
    
def edgeDetect(image,minVal=100,maxVal=200,aperature=3,key='d'):
    if(key==ord(',')):
        minVal = minVal-1
    elif(key==ord('.')):
        minVal = minVal+1
    elif(key==ord('<')):
        maxVal = maxVal-1
    elif(key==ord('>')):
        maxVal = maxVal+1
    elif(key==ord('a')):
        aperature=aperature-1
    elif(key==ord('A')):
        aperature=aperature+1
    elif(key==ord('d')):
        minVal=100
        maxVal=200
        aperature=3
    
    if(minVal<0):
        minVal = 0
    if(maxVal>255):
        maxVal = 255
    if(minVal>maxVal):
        minVal=maxVal-1
    if(aperature<0):
        aperature=0

    
    # Write some Text
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (20,300)
    fontScale              = 0.75
    fontColor              = (255,0,0)
    lineType               = 2

    infoString = "(m:{m},M:{M},a:{a})".format(m=minVal,M=maxVal,a=aperature)


    edges = cv2.Canny(image,minVal,maxVal,aperature)

    # edges = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #                                       cv2.THRESH_BINARY, minVal//2*2+1, aperature//2*2+1) 

    edgesWithInfo = cv2.putText(cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB),
        infoString,
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

    return [edgesWithInfo,[minVal,maxVal,aperature]]


def computeContrast(rawImg,blkImg,ax,ax1):
    
    t1 = np.clip(blkImg[:240,:324] - rawImg[:240,:324],0,1500)
    t2 = np.clip(blkImg[:240,324:] - rawImg[:240,324:],0,1500)
    np.seterr(divide='ignore')
    contrast = np.divide(t1-t2,t1+t2)
    contrast[np.isnan(contrast)]=0
    contrast[np.isinf(contrast)]=0
    
    buf = io.BytesIO()
    ax.cla()
    ax1.cla()

    ax1.grid('minor')
    # ax.set_ylim(0,3)
    # ax.set_xticks(np.arange(-1,1.1,0.1))
    ax1.set_yticks(np.arange(0,1.1,0.1))

    ax.hist(contrast.flatten(), bins=33,alpha=0.5, density=True, facecolor='red')
    ax1.hist(contrast.flatten(),bins=33,alpha=0.5, density=True, cumulative=True,\
        histtype='step',linewidth=2,facecolor='blue')

    plt.savefig(buf,format='png')
    buf.seek(0)
    histimg = cv2.imdecode(np.frombuffer(buf.read(),dtype=np.uint8),1)

    # Write some Text
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (50,30)
    fontScale              = 0.6
    fontColor              = (0,0,255)
    lineType               = 2

    infoString = "fraction>90%:{:2.2f}  |  fraction<-90%:{:2.2f}  |  median:{:2.2f}".format(\
        np.count_nonzero(contrast>=0.9)/contrast.size,\
        np.count_nonzero(contrast<=-0.9)/contrast.size,\
        np.median(contrast))


    imgWithInfo = cv2.putText(histimg,
        infoString,
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

    return imgWithInfo

def adjustExposure(camera,currentExp,key,numRep=16):
    minExp = int((26195*numRep +10)/1000) + 1
    sleepTime = 1/20
    if((key==ord('e')) or (key==ord('E'))):
        time.sleep(sleepTime)
        camera.Pause()

        if(key==ord('e')):
            currentExp=currentExp-1
        elif(key==ord('E')):
            currentExp=currentExp+1

        if(currentExp<minExp):
            currentExp = minExp

        time.sleep(sleepTime)
        camera.SetExposure(currentExp)
        print("New Exposure: {}".format(currentExp))

        camera.Resume()
        camera.readout_reset()
    return currentExp