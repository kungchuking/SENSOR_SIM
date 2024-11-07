
#!/usr/bin/env python3
import os
import sys
import numpy as np
import time
import cv2
from PIL import Image 
import matplotlib.pyplot as plt
import io
import atexit
from datetime import datetime
sys.path.insert(0, "./api")
from t6 import *
from usefulFunctions import *


[row, col] = [480, 680] #sensor resolution



cmap_heat = plt.get_cmap('inferno')
# hdr2ldr image
mergeMertens = cv2.createMergeMertens()
mergeMertens.setContrastWeight(1)
mergeMertens.setContrastWeight(1)
def hdr2ldr(img,vmin=0,vmax=1,LSB=4):
    """
        img:        2D np.uint16
        vmin,vmax:  scaling output images even further
        LSB:        What is the noise-free LSB in uint16. 
                    The 16-bit input image will be divide into (16-8-LSB+1)x 8-bit images
    """
    LSB    = int(LSB)
    nuImgs = 16-8-LSB+1 # number of 8-bit images

    # split 16-bit image into 8-bit images
    imgs = np.zeros((nuImgs,img.shape[0],img.shape[1]),dtype=np.uint8)
    for i in range(nuImgs):
        imgs[i,:,:] = np.interp(img,[0,2**(i+8+LSB)-1],[0,2**8-1]).astype(np.uint8)

    # fuse images to create hdr image
    fusion = mergeMertens.process(imgs[:,:,:col])
    fusion = np.interp(fusion,[vmin,vmax],[0,2**16-1]).astype(np.uint16)
    return fusion


def nothing(umm):
    """
        doesn't do anything. Need this for some image trackbar thingy
    """

    pass

def getADCs(cam, rows, NSUB, adc1_en=True, adc2_en = True):
    """
        cam: camera class handle
        returns: 2d array of size 320x648
    """
    

    if(adc2_en):
        # time222 = time.time()
        # print('time222',time222)
        raw_adc2   = cam.adc2_read(NSUB)
        # time11 = time.time()
        # print('time11', time11)
    else:
        raw_adc2 = None

    if(adc1_en):
        bt=t6.wire_out(0x24)
        print("state0: "+str(bt))
        raw_adc1   = cam.imread()
        # time111 = time.time()
        # print('time111', time111)
    else:
        raw_adc1 = None
    
    
    return  raw_adc1,raw_adc2
    # return raw

def arrangeImg(cam,raw_adc1=None,raw_adc2=None,rows_adc1=480,rows_adc2=480,adc2_PerCh=16):
    if(raw_adc1!=None):
        img_adc1 = cam.arrange_raw_T7(raw_adc1,rows_adc1)
    else:
        img_adc1 = np.zeros((row,col*2),dtype=np.uint16)
    if(raw_adc2!=None):
        img_adc2 = (1-cam.arrange_adc2(raw_adc2,rows=rows_adc2//freq * cluster,ADCsPerCh=adc2_PerCh))*(2**16-1)
    else:
        img_adc2 = np.zeros((row,col),dtype=np.uint16)

    return img_adc1,img_adc2

def getImg(cam,rows):
    """
        cam: camera class handle
        returns: 2d array of size 320x648
    """
    # tap,colsPerADC,ADCsPerCh,ch,nob = 2,2,20,12,12
    # cam.frame_length   = int(rows*tap*colsPerADC*ADCsPerCh*ch*nob/8*256/240)
    tap,colsPerADC,ADCsPerCh,ch,nob = 1,2,20,17,12
    cam.frame_length   = int(rows*tap*colsPerADC*ADCsPerCh*ch*nob/8*256/255)
    # cam.frame_length = 108*4*320*2*2 # t6
    # cam.frame_length = int((480*40*17*2*12*256/255) // 8) #t7

    raw              = cam.imread()
    # raw_img          = cam.arrange_raw(raw,rows)
    # tic = time.time()
    raw_img          = cam.arrange_raw_T7(raw,rows)
    # logging.info('rearrange time: {}'.format(time.time()-tic))
    return raw, raw_img    

def getSubImg(cam,NSUB):
    pread=bytearray(int(680*480*256/255*NSUB/8));
    t6.read(0xa3,pread)

    mean = (1-cam.arrange_adc2(pread))*(2**16-1)

    return mean.astype(np.uint16)

def showImg(win,img,cam=None,show=True,\
            raw=False,gain=False,black=False,dynamic=False,hdr=False,\
            crop = False, crop_loc = [175,235,185,245], \
            heatmap=False,drawLines=False,edgeDetectFlag=False,f=1,max_scale=2**13-1):
    """
        can return and/or show raw, black, gain, dynamic calibrated images

        win: name of the cv2 window where image should be showed
        img: np.uint16 2d (r x c) or 3d array of size (N x r x c)
        cam: camera handle

            FLAGS
        show:       show image in win
        raw:        don't scale show as it is
        gain:       do gain calibration
        black:      do black calibration
        dynamic:    dynamically adjust black and white levels
        heatmap:    gray to heatmap
        hdr:        combine 16-bit image to show on 8-bit display
        crop:       crop the image

            Optional Values/Handles
        f:          scale the image
        max_scale:  maxium allwed pixel value(saturation value) after black calibration
        trackbar:   trackbar name
        drawLines:  draw horizontal lines at 25, 50 and 75% of the image height
        crop_loc: location where to crop
    

        output
        img: the image being showed
    """
    if(raw):
        # cv2.imshow(win,img)
        # img = img
        img = cam.image_scale6(img,gain=False, black=False, max_scale=max_scale)
    elif(gain):
        img = cam.image_scale6(img,gain=gain, black=black, max_scale=max_scale)
    elif(black):
        img = cam.image_scale6(img,gain=False,black=True,max_scale=max_scale)
    elif(dynamic):
        img = cam.image_scale6(img,gain=False,black=False,dynamic=True)
    elif(hdr):
        LSB = np.floor(np.log2(2**16/max_scale)) #LSB without noise
        img = hdr2ldr(img,vmin=0,vmax=1,LSB=LSB)
    elif(heatmap):
        # re-formatting array to make sure we can use it for 2d arrays as well
        img = img.reshape((-1,img.shape[-2],img.shape[-1]))
        img = np.interp(img,[0,2**16-1],[0,1])
        img_combined = np.mean(img,axis=0)

        #apply colormap
        img = cmap_heat(img_combined)[:,:,[2,1,0]]
        img = np.interp(img,[0,1],[0,2**16-1]).astype(np.uint16)

    if(edgeDetectFlag):
        [img,stuff] = edgeDetect(img,minVal=100,maxVal=200,aperature=3,key='d')


    if(drawLines):
        if(len(img.shape)==2):
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR) #add RGB channels to grayscal image
        img[:, ::32   ,2] = 2**16-1
        # img[:, 1::32  ,2] = 2**16-1
        # img[:, 39::32 ,2] = 2**16-1
        # img[(row*2)//4, : ,1] = 2**16-1
        # img[(row*3)//4, : ,2] = 2**16-1

    if(crop):
        if(len(img.shape)==2):
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR) #add RGB channels to grayscal image
        [r1,c1,r2,c2] = crop_loc
        img[r1:r2,       [c1,c2],     2] = 2**16-1
        img[[r1,r2],      c1:c2,      2] = 2**16-1
        img[r1:r2,   [c1+col,c2+col], 2] = 2**16-1
        img[[r1,r2],  c1+col:c2+col,  2] = 2**16-1

        cropped_img = np.concatenate([img[r1-1:r2+1,c1-1:c2+1],img[r1-1:r2+1,c1+col-1:c2+col+1]],axis=1)

    if(show):
        cv2.imshow(win,cv2.resize((img),None,fx=f,fy=f,\
        # interpolation=cv2.INTER_LINEAR))
        interpolation=cv2.INTER_NEAREST))
        if(crop):
            cv2.imshow(win+'(cropped)',cv2.resize((cropped_img),None,fx=f*10,fy=f*10,\
            interpolation=cv2.INTER_NEAREST))
    

    return img


def exit_handler(cam):
    """
        Add code here that you want to execute when you close the camera.
        For example, you may want to print the parameters you updated 
            during camera operation.
    """

    # -- cam.dev.Close()
    # In this function, all the lines below this are optional
    # -- os.system('pkill feh') 
    # -- os.system('python3 ~/customErrorScript.py') #this is optional

    print("EXIT SCRIPT")
    print("exposure-----------: {}".format(exposure))
    print("rows_sub_img-------: {}".format(rows_sub_img))
    print("adc2_spacing-------: {}".format(adc2_spacing))





if __name__ == '__main__':

    # ====================================
    # Users can change these variables
    # Variables to play with for baby users
    # maskfile    = 'maskfile/t6_cats_256.bmp'
    # maskfile    = 'maskfile/t7_allOneZero_100x.bmp'
    # maskfile    = 'maskfile/t7_allOne_600x.bmp'
    # maskfile    = 'maskfile/t7_gradient_masks_256.bmp'
    # maskfile    = 'maskfile/t7_triangle_50x.bmp'
    # maskfile    = 'maskfile/t7_all1_16x.bmp'
    # maskfile    = 'maskfile/t7_mask_test.bmp'
    # maskfile    = 'maskfile/AllOne_100x.bmp'
    # maskfile    = 'maskfile/T7_all_zeroes.bmp'
    # maskfile    = 'maskfile/T7_padded_varstrengthtriangle_100x.bmp'
    # maskfile    = 'maskfile/T7_row_One_Zero.bmp'
    # maskfile    = 'maskfile/T7_Ones_100x_512.bmp'
    # maskfile    = 'maskfile/T7_Zeros_100x_512.bmp'
    # maskfile    = 'maskfile/T7_shapes_1024.bmp'
    # maskfile    = 'maskfile/output_tiled.bmp'
    # maskfile    = 'maskfile/triple_combined_output.bmp'
    #   maskfile = 'maskfile/output_1bit.bmp'
    # maskfile = './maskfile/T7_AllOne_AllZero_2x_4rep.bmp'
    # maskfile = './maskfile/T7_AllZero_100x.bmp'
    # maskfile = './maskfile/T7_AllOne_100x.bmp' 
    # maskfile = './maskfile/Alternate_AllOne_AllZero.bmp' 
    # maskfile    =   './maskfile/T7_50row_AllZero_50row_AllONe_100X.bmp'
    # maskfile    = 'maskfile/neutral.bmp'
    # maskfile    = 'maskfile/gradient.bmp'
    # maskfile    =   'maskfile/Testmask_50_allZero_50_halfrow_zero.bmp'
    maskfile = 'maskfile/burst_mask_10x10_rep1_tap1HS.bmp'
    if(0):
        subFrameNum = 100           # Number of subframes
        # exposure    = 53.85+0.08*60         # =(exposure time(us) per subframe)/2. must be larger than 26.2*repNum. Sorry it's kinda weird right now
        exposure    = 60         # =(exposure time(us) per subframe)/2. must be larger than 26.2*repNum. Sorry it's kinda weird right now
        numSubRO    = 870
        rows_sub_img = int(60)
        adc2_PerCh  = 20

        rows_test    = 480
        rows_masking = 60
        
        adc2_spacing        = 4718
        adc2_spacing_step   = 10
        row_start           = 0
        trigWaitTime        = 8172
    if(0):
        subFrameNum = 121           # Number of subframes
        exposure    = 278         # =(exposure time(us) per subframe)/2. must be larger than 26.2*repNum. Sorry it's kinda weird right now
        numSubRO    = 120
        rows_sub_img = int(480)
        adc2_PerCh  = 20

        rows_test    = 480
        rows_masking = 480
        
        adc2_spacing        = 55000
        adc2_spacing_step   = 10
        row_start           = 0
        trigWaitTime        = 10

    if(0):
        subFrameNum = 100          # Number of subframes
        exposure    = 100         # =(exposure time(us) per subframe)/2. must be larger than 26.2*repNum. Sorry it's kinda weird right now
        numSubRO    = 100
        rows_sub_img = int(480)
        adc2_PerCh  = 20

        rows_test    = 480
        rows_masking = 480
        
        adc2_spacing        = 55000
        adc2_spacing_step   = 10
        row_start           = 0
        trigWaitTime        = 10

    if(1):
        subFrameNum = 100 #8          # Number of subframes
        exposure    = 555       # =(exposure time(us) per subframe)/2. must be larger than 26.2*repNum. Sorry it's kinda weird right now
        numSubRO    = 100 #80 #8
        rows_sub_img = int(480)
        adc2_PerCh  = 20

        rows_test    = 480
        rows_masking = 480
        
        adc2_spacing        = 110438#110000#55000
        adc2_spacing_step   = 10
        row_start           = 0
        trigWaitTime        = 122710 #22710

    MASTER_MODE = True          # Running camera without a trigger. Set it to false to run it based on trigger

    # ---------------------------------    
    # Variables to play with for bigger baby users
    # bitfile     = "bitfile/Reveal_Top_t7_based_on_t6_test_02.bit" # CLKMi 100 MHz, exposure module changed back to t6
    # bitfile       = "bitfile/Reveal_Top_t7_based_on_t6_06.07.bit"
    # bitfile       = "bitfile/Reveal_Top_t7_based_on_t6_06.08.bit"
    # bitfile         =   "bitfile/Reveal_Top_10_07_23_T7_v11_ro.bit"
    # bitfile         =   "bitfile/Reveal_Top_10_07_23_T7_v12_ro.bit"
    # bitfile = "bitfile/Reveal_Top_18_07_23_T7_v13_quick.bit"
    # bitfile = "bitfile/Reveal_Top_18_07_23_T7_v13_extratimingopt.bit"
    # bitfile = "bitfile/Reveal_Top_18_07_23_T7_v13_retiming.bit"
    # bitfile = "bitfile/Reveal_Top_18_07_23_T7_v13_postroutephys.bit"
    # bitfile = "bitfile/Reveal_Top_18_07_23_T7_v13_default.bit"
    # bitfile = "bitfile/Reveal_Top_18_07_23_T7_v13_rowaddtest.bit"
    # bitfile = "bitfile/Reveal_Top_18_07_23_T7_v13_rowadd_ro_tst.bit"
    # bitfile = "bitfile/Reveal_Top_18_07_23_T7_v13_rowadd_dec_tst.bit"
    # bitfile = "bitfile/Reveal_Top_18_07_23_T7_v13_rowadd_dec_tst_all_hold.bit"
    # bitfile = "bitfile/Reveal_Top_debug.bit"
    # bitfile = "bitfile/Reveal_Top_ROMergeV2.bit"
    # bitfile = "bitfile/Reveal_Top_31_07_23_T7_v14_exp_ro_test_11.bit"
    # bitfile = "bitfile/Reveal_Top_09_08_23_T7_v14_exp_ro_1024_test_12.bit" # old ros
    # bitfile = "bitfile/Reveal_Top_09_08_23_T7_v14_exp_ro_1024_test_13.bit"
    # bitfile = "bitfile/Reveal_Top_09_08_23_T7_v14_exp_ro_1024_test_15.bit"
    # bitfile = "bitfile/Reveal_Top_09_08_23_T7_v14_col_prech_1024_test_17.bit"
    # bitfile = "bitfile/Reveal_Top_CEP_FDC_combo_v01.bit"
    # bitfile = "bitfile/Reveal_Top_exp6_new.bit"
    # bitfile = "bitfile/Reveal_Top_RO.bit"
    # bitfile = "bitfile/Reveal_Top_exro_adc2.bit"
    # bitfile = "bitfile/Reveal_Top_old_v2.bit"
    # bitfile = "bitfile/Reveal_Top_led_clk.bit"
    # bitfile = "bitfile/Reveal_Top_led_combined.bit"
    # bitfile = "bitfile/Reveal_Top_ADC2_fixed.bit"
    # bitfile = "bitfile/Reveal_Top_new.bit"
    # bitfile = "bitfile/Reveal_Top_adc1_test.bit"
    # bitfile = "bitfile/Reveal_Top_flag.bit"
    # bitfile = "bitfile/11.bit"
    # bitfile = "bitfile/ddr3-test.bit"
    # bitfile = "bitfile/test.bit"
    # bitfile = "bitfile/test-new.bit"
    # bitfile = "bitfile/new.bit"
    # bitfile = "bitfile/test.bit"
    # bitfile = "bitfile/timing.bit"
    # bitfile = "bitfile/test10.bit"
    # bitfile = "bitfile/testtest2.bit"
    # bitfile = "bitfile/test1.bit"
    # bitfile = "bitfile/ddr3test.bit"
    # bitfile = "bitfile/t1.bit"
    # bitfile = "bitfile/ddr3_priority.bit"
    # bitfile = "bitfile/ddr3test5.bit"
    # bitfile = "bitfile/ddr3_working.bit"
    # bitfile = "bitfile/dacV10FIFO.bit"
    # bitfile = "bitfile/row_testing.bit"
    # bitfile = "bitfile/led_2.bit"
    bitfile = "bitfile/integrated.bit"
    bitfile = "bitfile/Reveal_Top_TAP_programmable_v02.bit"
    repNum        = 1             # Number of repetition per subframe
    memMaskNum    = subFrameNum   # Number of masks in the memory
    exposureRep   = 1             # number of projector scans before 1 readout
    exposure_step = 1          # minium step is 0.005


    # ====================================
    adc1_en = 1
    adc2_en = 1

    # ======Don't modify lines BELOW THIS=======
    # just some necessary stuff. 
    l_par={} #local parameters
    l_par['bitfile'] = bitfile
    l_par['maskfile'] = maskfile
    l_par['subFrameNum'] = subFrameNum
    l_par['repNum'] = repNum
    l_par['memMaskNum'] = memMaskNum
    l_par['exposure'] = exposure
    l_par['exposureRep'] = exposureRep
    l_par['row'] = row
    l_par['col'] = col
    l_par['exposure_step'] = exposure_step

    # ---------------------------------    
    cnt = 0
    # Disable logging
    logging.disable(logging.DEBUG)
    logging.info("Disabling debug logging")

    # ---------------------------------    
    freq = 1
    start_row = 0
    cluster = 1

    # initialize camera
    t6 = T6(bitfile) # program camera
    t6.param_set(t6.param['MU_NUM_ROW'],rows_masking)
    t6.param_set(t6.param['MASK_SIZE'],rows_masking*subFrameNum*repNum*16)

    t6.param_set(t6.param['IMG_SIZE'],int((rows_test*40*17*1*12*256/255)//32))

    t6.param_set(t6.param['SUB_IMG_SIZE'],int((rows_sub_img*2*adc2_PerCh*17*numSubRO*1*256/255)//32 // freq * cluster))
    t6.param_set(t6.param['UNIT_SUB_IMG_SIZE'],int((rows_sub_img*2*adc2_PerCh*17*1*1*256/255)//32  //freq * cluster))
    t6.param_set(t6.param['N_SUBREADOUTS'],int(numSubRO))
    t6.param_set(t6.param['row_freq'], freq)
    t6.param_set(t6.param['start_row'], start_row)
    t6.param_set(t6.param['cluster'], cluster)
    t6.param_set(t6.param['led'], 4)
    t6.unit_subframe_length = round(rows_sub_img*17*2*adc2_PerCh*256/255//8 // freq  * cluster)
    t6.adc2_container = bytearray(round(t6.unit_subframe_length*numSubRO)) 

    # set camera to be in MASTER or SLAVE mode
    if(MASTER_MODE): # to run camera in master mode. Does not wait for any external trigger
        # Select_val =np.packbits([
        #              0,0,adc1_en,0, 0,adc2_en,adc1_en,0,   # 31:24 # ADC1 => Select_val[29] = 1 for Allone mask and complement for adc2
        #              0,0,0,0, 0,0,0,0,   # 23:16
        #              0,0,0,0, 0,0,0,1,   # 15:8
        #              0,1,1,1, 0,0,1,1 ]) #  7:0
        # Select_val =np.packbits([
        #              0,0,1,0, 0,adc2_en,adc1_en,0,   # 31:24 # ADC1 => Select_val[29] = 1 for Allone mask and complement for adc2
        #              0,0,0,0, 0,1,0,0,   # 23:16
        #              1,0,1,0, 0,0,0,1,   # 15:8
        #              1,0,1,1, 0,0,0,1 ]) #  7:0              ADC1
        # Select_val =np.packbits([
        #              0,0,0,0, 0,adc2_en,adc1_en,0,   # 31:24 # ADC1 => Select_val[29] = 1 for Allone mask and complement for adc2
        #              0,0,0,0, 0,0,0,0,   # 23:16
        #              0,1,0,1, 0,0,1,0,   # 15:8
        #              1,1,0,1, 1,1,1,1 ]) #  7:0                ADC2
        # Select_val =np.packbits([
        #              0,0,1,0, 0,adc2_en,adc1_en,0,   # 31:24 # ADC1 => Select_val[29] = 1 for Allone mask and complement for adc2
        #              0,0,0,0, 0,1,0,1,   # 23:16
        #              1,0,1,1, 0,0,0,1,   # 15:8
        #              1,0,1,1, 0,0,0,1 ]) #  7:0              ADC1 check ch1_en and ch2_en
        # Select_val =np.packbits([
        #              0,0,0,0, 0,adc2_en,adc1_en,0,   # 31:24 # ADC1 => Select_val[29] = 1 for Allone mask and complement for adc2
        #              0,0,0,0, 0,0,0,1,   # 23:16
        #              0,1,1,0, 0,0,1,0,   # 15:8
        #              0,1,1,0, 0,0,1,0 ]) #  7:0             check state
        Select_val =np.packbits([
                     0,0,0,0, 0,adc2_en,adc1_en,0,   # 31:24 # ADC1 => Select_val[29] = 1 for Allone mask and complement for adc2
                     0,0,0,1, 0,0,0,0,   # 23:16
                     0,0,0,0, 0,0,1,1,   # 15:8
                     1,1,1,1, 1,1,1,1]) #  7:0             check state
        Select_val = np.sum(np.multiply(Select_val,[2**24,2**16,2**8,1])).astype(np.uint32)
        t6.param_set(t6.param['Select'], int(Select_val)) #free run mode
    else: # to run the camera in slave mode. It waits for trigger. 
            # 3.3V trigger must be connected to 
        t6.param_set(t6.param['Select'], 0x80000000) #free run mode

    TADC = 20
    
    t6.param_set(t6.param['ADC1_TADC'],     TADC)
    t6.param_set(t6.param['ADC1_T1'],       int(TADC)*2)
    t6.param_set(t6.param['ADC1_T2_1'],     0)#int(TADC)*17)#int(TADC)*8)   # > TADC*14
    t6.param_set(t6.param['ADC1_T2_0'],     0)#int(TADC)*19)#int(TADC)*5)
    t6.param_set(t6.param['ADC1_T3'],       int(TADC)*12)
    t6.param_set(t6.param['ADC1_T4'],       int(TADC)*12+2)     # > TADC*14
    t6.param_set(t6.param['ADC1_T5'],       2)                # > TADC*14
    t6.param_set(t6.param['ADC1_T6'],       1)    #  =1
    t6.param_set(t6.param['ADC1_T7'],       int(TADC)+2)   #  > TADC
    t6.param_set(t6.param['ADC1_T8'],       2)
    t6.param_set(t6.param['ADC1_T9'],       int(TADC)+2) # T8 + x; x > TADC
    t6.param_set(t6.param['ADC1_Tcolumn'],  int(TADC)*27) #T4 + T7*12 + T5
    t6.param_set(t6.param['ADC1_NUM_ROW'],  rows_test)

    t6.param_set(t6.param['ADC2_NUM_ROW'],  int(rows_sub_img ))
    t6.param_set(t6.param['ADC2_Wait'],     100)
    t6.param_set(t6.param['ADC2_Tcolumn'],  50) #T4 + T7*12 + T5
    t6.param_set(t6.param['ADC2_T2_1'],     18)
    t6.param_set(t6.param['ADC2_T2_0'],     20)
    t6.param_set(t6.param['ADC2_T7'],       3+adc2_PerCh)
    t6.param_set(t6.param['ADC2_T8'],       2)
    t6.param_set(t6.param['ADC2_T9'],       2+adc2_PerCh)

    # t6.param_set(t6.param['ADC2_Wait'],     1)
    # t6.param_set(t6.param['ADC2_Tcolumn'],  560) #T4 + T7*12 + T5
    # t6.param_set(t6.param['ADC2_T2_1'],     3)
    # t6.param_set(t6.param['ADC2_T2_0'],     8)
    # t6.param_set(t6.param['ADC2_T7'],       26)
    # t6.param_set(t6.param['ADC2_T8'],       2)
    # t6.param_set(t6.param['ADC2_T9'],       22)
    
    t6.param_set(t6.param['MSTREAM_Select'],0xAAAAAAAA)

    # t6.param_set(t6.param['TrigOnTime'],)
    trigOffTime = adc2_spacing
    t6.param_set(t6.param['TrigOffTime'],    trigOffTime)
    t6.param_set(t6.param['TrigWaitTime'],   trigWaitTime)
    t6.param_set(t6.param['TrigNo'],        numSubRO)

    

    t6.param_set(t6.param['T_DEC_SEL_0'],   2  )
    t6.param_set(t6.param['T_DEC_SEL_1'],   0  )
    t6.param_set(t6.param['T_DEC_EN_0'],    4 )
    t6.param_set(t6.param['T_DEC_EN_1'],    0 )
    t6.param_set(t6.param['T_DONE_1'],2)

   

    t6.param_set(t6.param['Tdes2_d'],2)
    t6.param_set(t6.param['Tdes2_w'],4)
    t6.param_set(t6.param['Tmsken_d'],7)
    t6.param_set(t6.param['Tmsken_w'],10)
    
    # t6.param_set(t6.param['Tgl_Res'],10)


    t6.param_set(t6.param['ROWADD_INCRMNT'], row_start)

    
    atexit.register(exit_handler,t6)
    # t6.UploadMaskDummy(maskfile,memMaskNum)
    t6.UploadMask(maskfile,memMaskNum)
    t6.SetMaskParam(subFrameNum, repNum)
    t6.SetExposure(exposure)
    # t6.param_set(t6.param['NumExp'],int(exposureRep))

    # reset, do one exposure and wait till images are readout
    t6.readout_reset()
    time.sleep(1)
    # ======Don't modify lines ABOVE THIS=======


   # create imshow windows
    raw ='C2B RAW' ;cv2.namedWindow(raw)
    # cv2.createTrackbar('Zoom', raw, 5, 40, nothing); f=1

    blackCal ='C2B' ;cv2.namedWindow(blackCal)
    #hdr='C2B_(HDR2LDR)';cv2.namedWindow(hdr) # this may or may not work well


    saveFlag = False # set flag to save the image. Can be set by pressing 's' during camera operation

    # it will try to create following directory to save images if it doesn't existss
    saveDir = os.path.join('./image/t6Exp{:06d}Mask{:03d}'.format(\
                                int(exposure),subFrameNum),'')
    saveDir2 = os.path.join('./testimage/t6Exp{:06d}Mask{:03d}'.format(\
                                int(exposure),subFrameNum),'')
    # number of images to save
    saveNum         = 20 #20
    rawBuffer       = np.zeros((saveNum,row,col),        dtype=np.uint16) # buffer for raw images
    rawBuffer2       = np.zeros((saveNum,subFrameNum,row,col),        dtype=np.uint16)
    blackBuffer     = np.zeros((saveNum,row,col),        dtype=np.uint16) # buffer for black calibrated images
    # rawBuffer       = np.zeros((saveNum,row,col*2),        dtype=np.uint16) # buffer for raw images
    # blackBuffer     = np.zeros((saveNum,row,col*2),        dtype=np.uint16) # buffer for black calibrated images
    raw_adc1_buffer = np.zeros((saveNum, t6.frame_length), dtype=np.uint8)
    raw_adc2_buffer = np.zeros((saveNum, numSubRO*t6.unit_subframe_length), dtype=np.uint8)
    # following 3 lines are to calcualte frame rate.
    # press 'f' to ~pay respect~ find current frame rate.
    FRAMENUM = 0
    prev_time = time.time()
    prev_fNum = FRAMENUM

    [r1,c1,r2,c2]     = [102, 233, 162, 293]
    t6.dac_setter()
    t6.volt_setter()
    time.sleep(2)
    
    # This is the loop to view sensor output as video.
    while True:
        
        if(saveFlag):
            showFlag = False
        else:
            showFlag = True
        # time1 = time.time()
        # print('time1', time1)
        raw_adc1,raw_adc2=getADCs(t6,row,NSUB=numSubRO,adc1_en=adc1_en,adc2_en=adc2_en)
        # time2 = time.time()
        # print('time2', time2)
        if(showFlag):
            cnt = cnt+1
            img_adc1,img_adc2=arrangeImg(t6,raw_adc1,raw_adc2,rows_adc1=rows_test,rows_adc2=rows_sub_img,adc2_PerCh=adc2_PerCh)

            quit()

            # time5 = time.time()
            # print('time5', time5)
            print(cnt)
            f=cv2.getTrackbarPos('Zoom',raw)/5 #image scale factor
            if(adc1_en & adc2_en):
                raw_img = showImg(raw, cam=t6, show=True,
                            img=img_adc1, raw=True, black=False, dynamic=False, max_scale=8192, f=f)
                
                blackCal_img = showImg(blackCal, cam=t6, show=True,
                                       img=img_adc1, black=True, dynamic=False, gain=False,
                            # img=img_adc1, black=False, dynamic=True,
                            # img=raw_img, raw=False, black=False, dynamic=True, 
                            # img=2**16-1-raw_img*16, raw=True, black=False, dynamic=False,
                            # drawLines = True,
                            # crop_loc = [r1,c1,r2,c2], crop = True, 
                            max_scale=2048,f=f)
                showImg("subframes",img=img_adc2, cam=t6, heatmap=True)
            
            elif(adc1_en):
                raw_img = showImg(raw, cam=t6, show=True,
                            img=img_adc1, raw=True, black=False, dynamic=False, max_scale=8192, f=f)
                
                blackCal_img = showImg(blackCal, cam=t6, show=True,
                                       img=img_adc1, black=True, dynamic=False, gain=False,
                            # img=img_adc1, black=False, dynamic=True,
                            # img=raw_img, raw=False, black=False, dynamic=True, 
                            # img=2**16-1-raw_img*16, raw=True, black=False, dynamic=False,
                            # drawLines = True,
                            # crop_loc = [r1,c1,r2,c2], crop = True, 
                            max_scale=2048,f=f)
                # Save the numpy array to a .npy file
            
                np.save('./image/combined_array.npy', blackCal_img)
            elif(adc2_en):
                showImg("subframes",img=img_adc2, cam=t6, heatmap=True)
                # time6 = time.time()
                # print('time6', time6)
        FRAMENUM += 1 # For measuring FPS

        # do something based on inputs
        key = cv2.waitKey(1)
            ### CLOSE ###
        if key==27: # this is ESC key
            cv2.destroyAllWindows()
            t6.dev.Close()
            break
            ### SAVE ###
        elif(key==ord('s')): # save frames
            logging.info("Saving #{} images".format(saveNum))
            saveFlag=True;saveIndex = 0;blackUpdate=False;brightUpdate=False
        elif(key==ord('b')): # do black calibration
            logging.info("Doing black calibration.\nIf lens not covered cover and do this again")
            saveFlag=True;saveIndex = 0;blackUpdate=True;brightUpdate=False
        elif(key==ord('v')): # capture bright image
            logging.info("Doing bright calibration.\nIf not showing uniform image, fix the scene and try again")
            saveFlag=True;saveIndex = 0;blackUpdate=False;brightUpdate=True

        elif key==ord('r'): # readout reset
            t6.readout_reset()

            ### ADC2 SPACING TIME ###
        elif key==ord('j'): # decrease adc2 spacing time
            time.sleep(0.1)
            adc2_spacing -= adc2_spacing_step
            t6.param_set(t6.param['TrigOffTime'],   int(adc2_spacing))
            print(adc2_spacing)
        elif key==ord('k'): # increase adc2 spacing time
            time.sleep(0.1)
            adc2_spacing += adc2_spacing_step
            t6.param_set(t6.param['TrigOffTime'],   int(adc2_spacing))
            print(adc2_spacing)
        elif key==ord('m'): # resize exposure change step
            try:
                adc2_spacing_step = float(input('Current adc2 spacing : {}\n\
                                           \rCurrent step size    : {}\n\
                                           \rnew step step        :'.format(\
                                adc2_spacing, adc2_spacing_step)))
            except:
                logging.info('m did not work. Try again')

        elif key==ord('g'): # decrease trigger delay
            trigWaitTime -= adc2_spacing_step
            logging.info('TrigWaitTime:{}'.format(int(trigWaitTime)))
            t6.param_set(t6.param['TrigWaitTime'],   int(trigWaitTime))

        elif key==ord('h'): # increase trigger delay
            trigWaitTime += adc2_spacing_step
            logging.info('TrigWaitTime:{}'.format(int(trigWaitTime)))
            t6.param_set(t6.param['TrigWaitTime'],   int(trigWaitTime))

        elif key==ord('o'): # Increment Row start address by 30 #ayandev
            row_start += 40
            row_start = min(row_start,480-40)
            logging.info('Row address incremented by : {}'.format(row_start))
            t6.param_set(t6.param['ROWADD_INCRMNT'],  int(row_start))

        elif key==ord('p'): # decrement Row start address by 30
            row_start -= 40
            row_start = max(row_start,0)
            logging.info('Row address decremented by : {}'.format(row_start))
            t6.param_set(t6.param['ROWADD_INCRMNT'],  int(row_start))


            ### EXPOSURE TIME ###
        elif key==ord('w'): # decrease exposure time
            if(exposure>exposure_step):
                exposure -= exposure_step
                exposure = max(exposure,26.23)
            t6.SetExposure(exposure)
            l_par['exposure'] = exposure
        elif key==ord('e'): # increase exposure time
            exposure += exposure_step
            t6.SetExposure(exposure) 
            l_par['exposure'] = exposure
            
        elif key==ord('d'): # resize exposure change step
            exposure_step = float(input('Current exposure value: {}\n\
                                     \rCurrent exposure step : {}\n\
                                     \rnew exposure step     :'.format(\
                        exposure,exposure_step)))
            l_par['exposure_step'] = exposure_step

            ### SHOW FRAME RATE ###
        elif(key==ord('f')): # press f to show ~~respect~~ frame rate
            new_time = time.time()            
            logging.info("fram#: {} time: {} fps: {}".format(\
                FRAMENUM, new_time, (FRAMENUM - prev_fNum)/(new_time-prev_time)))
            prev_time = time.time()
            prev_fNum = FRAMENUM


        elif(key==81):
            if(c1>=0):
                c1-=1;c2-=1
        elif(key==83):
            if(c2<480):
                c1+=1;c2+=1
        elif(key==82):
            if(r1>=0):
                r1-=1;r2-=1
        elif(key==84):
            if(r2<360):
                r1+=1;r2+=1

            ### SAVE ###
            # If saving, load images in buffer before saving
        if(saveFlag==True):
            if(adc1_en):
                raw_adc1_buffer[saveIndex,:]    = np.frombuffer(raw_adc1,dtype=np.uint8)
            if(adc2_en==1 and not(blackUpdate)):
                raw_adc2_buffer[saveIndex,:]    = np.frombuffer(raw_adc2,dtype=np.uint8)
            saveIndex+=1
            time.sleep(0.1)
            if(saveIndex == saveNum):
                saveIndex = 0
                saveFlag = False
                for i in range(saveNum):
                    raw_adc1 = bytearray(raw_adc1_buffer[i,:])
                    raw_adc2 = bytearray(raw_adc2_buffer[i,:])
                    
                    # img_adc1,img_adc2=arrangeImg(t6,raw_adc1,raw_adc2,rows_adc1=rows_test,rows_adc2=rows_sub_img,adc2_PerCh=adc2_PerCh)

                    if(adc1_en):
                        print('0')
                        rawBuffer[i,:,:] = img_adc1.copy()
                        blackBuffer[i,:,:] = showImg(raw, cam=t6, img=img_adc1, black=True, show=True,
                                            max_scale=2048,f=f)
                        if(not(blackUpdate)):
                            print('1')
                            # t6.imsave(blackBuffer[i,:,:], saveDir,'{:04d}.png'.format(i),full=False)
                            t6.imsave(rawBuffer[i,:,:], saveDir,'{:04d}.npy'.format(i))
                            cv2.imwrite(os.path.join(saveDir,'{:04d}.png'.format(i)),blackBuffer[i,:,:])
                            for ii in range(rawBuffer.shape[0]):
                                raw_file_name = '{:04d}.npy'.format(ii)
                                black_file_name = '{:04d}.png'.format(ii)

   
                                print(f"Raw buffer file name: {raw_file_name}")
                                print(f"Black buffer file name: {black_file_name}")



                    if(adc2_en==1 and not(blackUpdate)):
                        rawBuffer2[i,:,:,:] = img_adc2.copy()
                        t6.imsave(rawBuffer2[i,:,:,:],saveDir,'subbyte_row_{:04d}_{:04d}.npy'.format(row_start, i))

                        #t6.imsave(raw_adc2_buffer[i,:],saveDir,'subbyte_row_{:04d}_{:04d}.npz'.format(row_start, i))

                        # img_adc1,img_adc2=arrangeImg(t6,raw_adc1,raw_adc2,rows_adc1=rows_test,rows_adc2=rows_sub_img,adc2_PerCh=adc2_PerCh)
                        # heatmap_adc2 = showImg("subframes",img=img_adc2, cam=t6, heatmap=True, show=True)
                        # cv2.imwrite(os.path.join(saveDir,'subheat_row_{:04d}_{:04d}.png'.format(row_start, i)),heatmap_adc2)
                    # time.sleep(1/20)

                if(blackUpdate):
                    blackUpdate=False
                    np.save(t6.black_img_file,np.mean(rawBuffer[:,:,:],axis=0))
                    t6.black_img = np.load(t6.black_img_file)
                    logging.info("Completed black calibration")
                if(brightUpdate):
                    brightUpdate=False
                    np.save(t6.bright_img_file,np.mean(rawBuffer[:,:,:],axis=0))
                    t6.bright_img = np.load(t6.bright_img_file)
                    logging.info("Completed bright calibration")

                t6.imsave(t6.black_img, saveDir, 'black_img.npy'.format(i))
                t6.imsave(t6.bright_img,saveDir, 'bright_img.npy'.format(i))

                logging.info("Done saving")
