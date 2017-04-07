import sys
import numpy as np
import lutorpy as lua
try:
    import torchfile
except ImportError:
    torchfile = None
if torchfile is None:
    raise ImportError("Please, do pip install torchfile.")
import os
import sys
import time
import cv2
TFEAT_PATCH_SIZE = 32
TFEAT_DESC_SIZE = 128
TFEAT_BATCH_SIZE = 1000
STATS_FNAME = 'nets/stats.liberty.t7'
MODEL_FNAME = 'nets/tfeat_liberty_margin_star.t7'

try:
    stats = torchfile.load(STATS_FNAME)
    MEAN = stats['mi']
    STD = stats['sigma']
except:
    print "Please, ensure that there is nets/stats.liberty.t7 file"
    sys.exit(1)

def preprocess_patch(patch):
    out = cv2.resize(patch, (TFEAT_PATCH_SIZE, TFEAT_PATCH_SIZE)).astype(np.float32) / 255;
    out = (out - MEAN) / STD
    return out.reshape(1, TFEAT_PATCH_SIZE, TFEAT_PATCH_SIZE)

def extract_tfeats(net,patches):
    num,channels,h,w = patches.shape
    patches_t = torch.fromNumpyArray(patches)
    patches_t._view(num, 1, TFEAT_PATCH_SIZE, TFEAT_PATCH_SIZE)
    patches_t = patches_t._split(TFEAT_BATCH_SIZE)
    descriptors = []
    for i in range(int(np.ceil(float(num) / TFEAT_BATCH_SIZE))):
        prediction_t = net._forward(patches_t[i]._cuda())
        prediction = prediction_t.asNumpyArray()
        descriptors.append(prediction)
    out =  np.concatenate(descriptors)
    return out.reshape(num, TFEAT_DESC_SIZE)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Wrong input format. Try ./extract_desciptors_from_hpatch_file.py imgs/ref.png ref.TFEAT')
        sys.exit(1)

    input_img_fname = sys.argv[1]
    output_fname = sys.argv[2]

    t = time.time()
    image = cv2.imread(input_img_fname,0) #hpatch image is patch column 65*n x 65
    h,w = image.shape
    n_patches = h/w
    print('{0} patches to describe in {1}'.format(n_patches, input_img_fname))
    patches = np.zeros((n_patches,1,TFEAT_PATCH_SIZE, TFEAT_PATCH_SIZE))
    for i in range(n_patches):
        patches[i,:,:,:] = preprocess_patch(image[i*(w): (i+1)*(w), 0:w]) 
    require('nn')
    require('cunn')
    require('cudnn')
    net = torch.load(MODEL_FNAME)
    print 'Initialization and preprocessing time', time.time() - t
    t = time.time()
    out_descs = extract_tfeats(net,patches)
    print 'extraction time', time.time() - t 
    np.savetxt(output_fname, out_descs, delimiter=' ', fmt='%10.7f')    
