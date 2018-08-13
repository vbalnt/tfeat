import sys
import cv2

import numpy as np
import lutorpy as lua

class LutorpyNet:
    """
    Wrapper to Torch for loading models
    """
    # TODO: check codes for a bigger batch size.
    # Now it only works with size of one.
    batch_sz = 5000
    # TFeat number of input channels
    input_channels = 1
    # TFeat image input size
    input_sz = 32
    # TFeat descriptor size
    descriptor_sz = 128

    def __init__(self, model_file):
        """
        Class constructor

        :param model_file: The torch file with the trained model
        """
        require('nn')
        require('cudnn')
        self.net = torch.load(model_file)
        self.ones_arr = np.ones((self.input_sz, self.input_sz), dtype=np.uint8)

    def rectify_patch(self, img, kp, patch_sz):
        """
        Extract and rectifies the patch from the original image given a keyopint

        :param img: The input image
        :param kp: The OpenCV keypoint object
        :param patch_sz: The size of the patch to extract

        :return rot: The rectified patch
        """
        # TODO: check this routine since it does not work at all

        s = 1.5 * float(kp.size) / float(patch_sz)

        c = 1.0 if (kp.angle < 0) else np.cos(kp.angle*np.pi/180)
        s = 0.0 if (kp.angle < 0) else np.sin(kp.angle*np.pi/180)

        M = np.array([[s*c, -s*s, (-s*c + s*s) * patch_sz / 2.0 + kp.pt[0]],
                      [s*s,  s*c, (-s*s - s*c) * patch_sz / 2.0 + kp.pt[1]]])

        rot = cv2.warpAffine(img, np.float32(M), (patch_sz, patch_sz), \
              cv2.WARP_INVERSE_MAP + cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS)

        return rot

    def extract_patches(self, img, kpts):
        """
        Extract the patches and subtract the mean

        :param img: The input image
        :param kpts: The set of OpenCV keypoint objects
        
        :return: An array with the patches with zero mean
        """
        patches = []
        for kp in kpts:
            # extract patch
            sub = cv2.getRectSubPix(img, (int(kp.size*1.3), int(kp.size*1.3)), kp.pt)
            #sub = self.rectify_patch(img, kp, self.input_sz)
  
            # resize the patch
            res = cv2.resize(sub, (self.input_sz, self.input_sz))
            # subtract mean
            nmean = res - (self.ones_arr * cv2.mean(res)[0])
            nmean = nmean.reshape(self.input_channels, self. input_sz, self.input_sz)
            patches.append(nmean)

        return np.asarray(patches)

    def compute(self, img, kpts):
        """
        Compute the descriptors given a set of keypoints

        :param img: The input image
        :param kpts: The set of OpenCV keypoint objects
        
        :return: An array the descriptors
        """
        # number of keypoints
        N = len(kpts)

        # extract the patches given the keypoints
        patches = self.extract_patches(img, kpts)
        assert N == len(patches)

        # convert numpy array to torch tensor
        patches_t = torch.fromNumpyArray(patches)
        patches_t._view(N, self.input_channels, self.input_sz, self.input_sz)

        # split patches into batches
        patches_t   = patches_t._split(self.batch_sz)
        descriptors = []

        for i in range(int(np.ceil(float(N) / self.batch_sz))):
            # infere Torch network
            prediction_t = self.net._forward(patches_t[i]._cuda())
            
            # Cast TorchTensor to NumpyArray and append to results
            prediction = prediction_t.asNumpyArray()

            # add the current prediction to the buffer
            descriptors.append(prediction)

        return np.float32(np.asarray(descriptors).reshape(N, self.descriptor_sz))

def compute_and_draw_homography(frame, img, good, kp1, kp2, descriptor):
    """
    Compute the homography between a set of 2D correspondences and draw it to a given image

    :param img: The input image
    :params good: The set of correspondences
    :param kp1: The set of OpenCV keypoint objects in the object image
    :param kp2: The set of OpenCV keypoint objects in the input image
    :param descriptor: A string with the name of the descriptor
    """
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h,w,d = img.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)

    if descriptor == 'TFEAT':
        colour    = (255,0,0)
        thickness = 3
        position  = (480, 50)
    else:
        colour    = (0,0,255)
        thickness = 1
        position  = (480, 90)

    img2 = cv2.polylines(frame, [np.int32(dst)], True, colour, thickness, cv2.LINE_AA)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img2, descriptor, position, font, 1, colour, 2)

    return img2, matchesMask

def main(torch_file, video_file, object_img):
    """
    Main routine

    :param torch_file: Path to the Torch model file
    :params video_file: Path to the input video file
    :param object_img: Path to object image file
    """
    # the required minimum number of good matches to 
    # compute the homography between the object image
    # and the current frame.
    MIN_MATCH_COUNT = 10

    # create CNN descriptor
    net = LutorpyNet(torch_file)

    # start opencv stuff

    cap = cv2.VideoCapture(video_file)
    cap = cv2.VideoCapture(0)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 10.0, (640,480))

    # initialise ORB detector
    n_kpts = 1000
    det = cv2.ORB_create(n_kpts)

    # initialise matcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # read object image and compute descriptors
    img = cv2.imread(object_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kp11  = det.detect(gray, None)
    des11 = net.compute(gray, kp11)
    kp21 = kp11
    _ , des21 = det.compute(gray, kp11)

    # video loop

    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # find the keypoints and the descriptors
        kp12  = det.detect(gray, None)
        des12 = net.compute(gray, kp12)
        kp22 = kp12
        _ , des22 = det.compute(gray, kp12)

        # match features
        good1 = bf.match(des11, des12) 
        good2 = bf.match(des21, des22) 

        # comptue the homography and it to the current frame

        if len(good1) > MIN_MATCH_COUNT:
            compute_and_draw_homography(frame, img, good1, kp11, kp12, 'TFEAT')
        else:
            print "Not enough matches are found with TFEAT - %d/%d" % (len(good1), MIN_MATCH_COUNT)

        if len(good2) > MIN_MATCH_COUNT:
            compute_and_draw_homography(frame, img, good2, kp21, kp22, 'ORB')
        else:
            print "Not enough matches are found with SIFT - %d/%d" % (len(good2), MIN_MATCH_COUNT)

        # save frame to video
        out.write(frame)

        # Display the resulting frame
        cv2.imshow('window', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything done, release the capture
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'Not enough arguments.\n' \
              'Usage: python tfeat_demo.py torch_file.t7 video_file.webm object_img.png'
        sys.exit(0)

    main(sys.argv[1], sys.argv[2], sys.argv[3]) 
