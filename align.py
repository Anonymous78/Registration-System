from __future__ import print_function
import sys
import cv2
import numpy as np
import os
import shutil
import random

import openface
import openface.helper
from openface.data import iterImgs

def alignMain(outputDir, inputDir, landmark="outerEyesAndNose"):
    """function that performs the alignment of faces in the input directory using the provided landmarks and writes the aligned faces to disk in the provided outputDir

    paramters: outputDir, inputDir, landmark
    type parameter: string, string, string

    return: aligned files are written to disk"""
    openface.helper.mkdirP(outputDir)

    imgs = list(iterImgs(inputDir))

    # Shuffle so multiple versions can be run at once.
    random.shuffle(imgs)

    landmarkMap = {
        'outerEyesAndNose': openface.AlignDlib.OUTER_EYES_AND_NOSE,
        'innerEyesAndBottomLip': openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP
    }
    if landmark not in landmarkMap:
        raise Exception("Landmarks unrecognized: {}".format(landmark))

    landmarkIndices = landmarkMap[landmark]

    align = openface.AlignDlib("shape_predictor_68_face_landmarks.dat")

    nFallbacks = 0
    for imgObject in imgs:
        print("=== {} ===".format(imgObject.path))
        outDir = os.path.join(outputDir, imgObject.cls)
        openface.helper.mkdirP(outDir)
        outputPrefix = os.path.join(outDir, imgObject.name)
        imgName = outputPrefix + ".png"

        if os.path.isfile(imgName):
            print("  + Already found, skipping.")
        else:
            rgb = imgObject.getRGB()
            if rgb is None:
                print("  + Unable to load.")
                outRgb = None
            else:
                outRgb = align.align(96, rgb,
                                     landmarkIndices=landmarkIndices,
                                     skipMulti=True)
                if outRgb is None:
                    print("  + Unable to align.")

            if outRgb is not None:
                print("  + Writing aligned file to disk.")
                outBgr = cv2.cvtColor(outRgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(imgName, outBgr)
