#!/usr/bin/env python2
from __future__ import print_function
import time
start = time.time()

import cv2
import os
import cPickle
import sys

from operator import itemgetter

import numpy as np
np.set_printoptions(precision=2)

import pandas as pd

import openface

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB




def train(workDir, classifier='DBN',ldaDim=-1):
    """
    Function that performs training on the image representation using the default classifier model. A different model can be specificed.

    parameter: image representation, classifier, ldaDim
    Return: pickle file

    """

    print('Loading embeddings')
    file_name = '{}/labels.csv'.format(workDir)
    labels = pd.read_csv(file_name, header=None).as_matrix()[:,1]
    labels = map(itemgetter(1),
                map(os.path.split,
                    map(os.path.dirname, labels)))#Gets the image directory
    file_name = "{}/reps.csv".format(workDir)
    embeddings = pd.read_csv(file_name, header=None).as_matrix()
    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)
    nClasses = len(le.classes_)
    print("Training for {} classes.".format(nClasses))

    if classifier == 'LinearSvm':
        clf = SVC(C=1, kernel='linear', probability = True)
    elif classifier == 'GaussianNB':
        clf = GaussianNB()
    elif classifier == 'DBN':
        from nolearn.dbn import DBN
        print(labelsNum[-1:][0] + 1, embeddings.shape)
        clf = DBN([embeddings.shape[1], 500, labelsNum[-1:][0] + 1],
                    learn_rates=0.3,
                    learn_rate_decays=0.9,
                    learn_rates_pretrain=0.005,
                    epochs=300,#No of iterations
                    minibatch_size=1)#for a small data size set minibatch_size to 1. Otherwise the default is 64
    if ldaDim > 0:
        clf_final = clf
        clf = Pipeline([('lda', LDA(n_components=ldaDim)),
                        ('clf', clf_final)])
    clf.fit(embeddings, labelsNum)

    file_name = '{}/classifier.pkl'.format(workDir)
    print("Saving classifier to '{}'".format(file_name))
    with open(file_name, 'w') as f:
        pickle.dump((le, clf), f)




# if __name__ == '__main__':
#     align = openface.AlignDlib("shape_predictor_68_face_landmarks.dat")
#     net = openface.TorchNeuralNet('nn4.small2.v1.t7','96')
#     getRepresentation('20170726_191008.jpg')
