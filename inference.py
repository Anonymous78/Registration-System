from __future__ import print_function
import time
import cv2
import cPickle
import sys
import numpy as np
np.set_printoptions(precision=2)

import openface
from sklearn.preprocessing import LabelEncoder

start = time.time()
align = openface.AlignDlib("shape_predictor_68_face_landmarks.dat")
net = openface.TorchNeuralNet('nn4.small2.v1.t7',imgDim=96, cuda=False)
print("Loading the dlib and Openface models took {} seconds".format(time.time()-start))

def getRepresentation(imgPath):
    """
    Function that generates the facial embeddings from the image that contains just a single face.

    imgPath: image file
    Return: sorted list

    """

    start = time.time()
    bgr_image = cv2.imread(imgPath)
    if bgr_image is None:
        raise Exception("Unable to load image: {}".format(imgPath))

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    print(" + Original size: {}".format(rgb_image.shape))
    print("Loading the image took {} seconds".format(time.time() - start))

    start = time.time()
    bbl = align.getLargestFaceBoundingBox(rgb_image)
    bbs = [bbl]
    if len(bbs) == 0 or bbl is None:
        raise Exception("Unable to find a face: {}".format(imgPath))
    print("Face detection took {} seconds".format(time.time()-start))

    reps = list()
    for bb in bbs:
        start = time.time()
        alignedFace = align.align('96',
                rgb_image,
                bb,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            raise Exception('Unable to align image: {}'.format(imgPath))
        print('Alignment took {} seconds'.format(time.time() - start))
        print('This bounding box is centered at ({0},{1})'.format(bb.center().x, bb.center().y))

        start = time.time()
        rep = net.forward(alignedFace)
        print('Neural network forward pass took {} seconds'.format(time.time() - start))

        reps.append((bb.center().x, rep))

    sreps = sorted(reps, key=lambda x:x[0])
    return sreps

def infer(classifierModel,img):
    """The funciton uses the classifierModel generated from the training dataset to try and predict whose image its 'sees'. The function returns the predicted person and the confidence of that prediction.

    type classifierModel parameter: pickle file
    type img parameter: string

    return type: tuple
    """
    with open(classifierModel, 'rb') as f:
        if sys.version_info[0] < 3:
            (le, clf) = pickle.load(f)
        else:
            (le, clf) = pickle.load(f, encoding='latin1')

    print("\n=== {} ===".format(img))
    reps = getRepresentation(img)

    rep = r[1].reshape(1,-1)#reshape the matrix into a 1 by unknown coloumn size. Hence the -1. Bounding box is in the first index
    predictions = clf.predict_proba(rep).ravel()#perfoms the prediction.ravel member function works the same as reshape(-1) to flatten a multidimensional array into a 1D array.
    max_index = np.argmax(predictions)
    person = le.inverse_transform(max_index)
    confidence = predictions[max_index]

    return (person.decode('utf-8'), confidence)
