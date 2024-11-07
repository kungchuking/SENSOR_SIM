import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import time

def arrange_raw_T7(image: bytearray, rows: int = 480, target: np.ndarray = None) -> np.ndarray:
    print ("[INFO] ---- ---- ")
    print ("[INFO] Invoking arrange_raw_T7() ... ")
    print ("[INFO] len(image): ", len(image))

    tics = [['img0',time.time()]]
    ##rows:480, col:680, taps:2
    taps, colsPerADC, ADCsPerCh, ch = 1, 2, 20, 17
    # taps, colsPerADC, ADCsPerCh, ch = 1, 2, 20, 17
    cols = colsPerADC * ADCsPerCh * ch

    # -- Convert a bytearray object into a numpy array.
    img = np.frombuffer(image,dtype=np.uint8)
    tics.append(['img1',time.time()])

    print ("[INFO] img.shape: ", img.shape)

    # -- Reshape the numpy array into one that of size Nx8x4
    # -- Flip the least significant 4 bytes.
    img2=img.reshape(-1,8,4)[:,:,::-1]

    tics.append(['img2',time.time()])

    img3=np.unpackbits(img2,axis=2).reshape(-1,256) # recover the bits as stored in fifo
    tics.append(['img3',time.time()])

    img4 = img3[:,1:] # get rid of padded bits

    print ("[INFO] img4.shape: ", img4.shape)

    #reorder columns from 15 split unevenly to all 20 in a row
    img4_1 = img4.reshape(-1,17,15)
    img4_1 = np.moveaxis(img4_1,0,1)

    print ("[INFO] img4_1.shape: ", img4_1.shape)

    img4_1 = img4_1.reshape(-1,(int(np.shape(img4_1)[1]*15/20)),20)
    
    print ("[INFO] img4_1.shape: ", img4_1.shape)

    img4_1 = np.moveaxis(img4_1,0,1)

    print ("[INFO] img4_1.shape: ", img4_1.shape)

    # -- Commented out by Chu King on 6th November 2024, as this step is redundant.
    # -- img4_1 = img4_1.reshape(-1,340)

    tics.append(['img4',time.time()])

    noch    = 17 # total number of channels
    nob     = 12 # number of bits per channel
    img5 = img4_1.reshape(-1,nob,noch,20)         # convert into 12 separate data channels

    print ("[INFO] img5.shape: ", img5.shape)

    tics.append(['img5',time.time()])

    # img6 = np.moveaxis(img5,[1,2,3],[3,1,2])[:,:,::-1]    
    img6 = np.moveaxis(img5,1,3)[:,:,:,::-1]

    print ("[INFO] img6.shape: ", img6.shape)

    tics.append(['img6',time.time()])

    # at this point
    # AXES: 
    #    0: Each conversion (#rows x #taps x #cols/ADC)
    #    1: each digital channel
    #    2: columns serialized by each channel
    #    3: the bits for each column

    # img7_shape       = np.array(img6.shape); img7_shape[-1] = 16
    # img7             = np.zeros(img7_shape,dtype=np.bool)
    # img7[:,:,:,-12:] = img6
    # pad_loc          = np.zeros((4,2),dtype=int);pad_loc[-1,0]=16-nob
    # img7             = np.pad(img6,[[0,0],[0,0],[0,0],[4,0]])
    img7    = img6
    tics.append(['img7',time.time()])
    img8    = np.packbits(img7,axis=-1).astype(np.uint16)

    tics.append(['img8_packbits',time.time()])
    # img8    = img8[:,:,:,0]*256+img8[:,:,:,1]
    img8      = (img8[:,:,:,0]<<4).astype(np.uint16) + (img8[:,:,:,1]>>4).astype(np.uint16)
    
    # col_map = np.arange(20)
    # col_map = [19,9,14,4,18,8,13,3,16,6,11,1,17,7,12,2,15,5,10,0]
    col_map = [0, 4, 12, 8, 16, 2, 6, 14, 10, 18, 1, 5, 13, 9, 17, 3, 7, 15, 11, 19]
    ch_map  = np.arange(17)

    print ("[INFO] img8.shape: ", img8.shape)

    img8    = img8[:,ch_map,:]
    img9 = img8[:,:,[col_map]].reshape(rows,taps,colsPerADC,ADCsPerCh*ch)

    print ("[INFO] img9.shape: ", img9.shape)

    tics.append(['img9',time.time()])

    img10 = np.zeros((rows,taps,colsPerADC*ADCsPerCh*ch),dtype=np.uint16)

    print ("[INFO] img10.shape: ", img10.shape)

    img10[:,:,0::2] =   img9[:,:,0,:]
    img10[:,:,1::2] =   img9[:,:,1,:]
    
    tics.append(['img10',time.time()])

    img11 = img10[:,0,:]
    # img11 = np.concatenate([img10[:,0,:],img10[:,1,:]],axis=1)
    tics.append(['img11',time.time()])
    
    img12 = img11

    plt.imshow(img11, cmap="gray")
    plt.show()

    img12 = np.zeros(img11.shape, dtype=np.uint16)
    img12[0::2] = img11[:240]
    img12[1::2] = img11[:240]
    img12[0::2] = img11[240:]
    img12[1::2] = img11[240:]

    print ("[INFO] img12.shape: ", img12.shape)
    quit()

    return img12

img = cv2.imread("./image.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = img[:480, :680]

img_even_col = img[:, 0::2]
img_odd_col  = img[:, 1::2]

col_map_ = [0, 4, 12, 8, 16, 2, 6, 14, 10, 18, 1, 5, 13, 9, 17, 3, 7, 15, 11, 19]
col_map  = [0 for _ in range(20)]
for i in range(20):
    col_map[col_map_[i]] = i

img_odd_col = img_odd_col.reshape(-1, 17, 20)[..., col_map]
img_even_col = img_even_col.reshape(-1, 17, 20)[..., col_map]

container = np.zeros((img_even_col.shape[0], 2, *img_even_col.shape[-2:]))

container[:, 0, ...] = img_even_col
container[:, 1, ...] = img_odd_col

container = container[..., np.newaxis]

# -- TODO
target = np.array(img)
print ("[INFO] target.shape: ", target.shape)
print ("[INFO] container.shape: ", container.shape)

container = np.unpackbits(container.astype(np.uint8), axis=-1)

# -- 8 to 12 bit conversion
container_ = np.zeros((*container.shape[:-1], 12))
container_[..., -8:] = container
container_ = container_.reshape(container_.shape[0] * container_.shape[1], *container_.shape[2:])

# -- Flip the 12 bits
container_ = container_[..., ::-1]
container_ = np.moveaxis(container_, source=-1, destination=1)
container_ = container_.reshape(container_.shape[0] * container_.shape[1], *container_.shape[2:])

print ("[INFO] container_.shape: ", container_.shape)

container_ = np.moveaxis(container_, source=1, destination=0)
container_ = np.reshape(container_, (container_.shape[0], -1, 15))
container_ = np.moveaxis(container_, source=1, destination=0)
container_ = np.reshape(container_, (container_.shape[0], -1))

print ("[INFO] container_.shape: ", container_.shape)

# -- Pad one more bit on the MSB.
_container_ = np.zeros((container_.shape[0], 256)).astype(np.uint8)
_container_[:, 1:] = container_
_container_ = np.packbits(_container_, axis=-1)

# -- Flip the bytes once every 32 bits
_container_ = np.reshape(_container_, (-1, 8, 4))
_container_ = _container_[..., ::-1]
_container_ = np.reshape(_container_, (-1, 32))

# -- Flatten
_container_ = _container_.flatten()

print ("[INFO] _container_.shape: ", _container_.shape)

arrange_raw_T7(_container_, target=target)


