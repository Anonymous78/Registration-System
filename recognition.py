#!/usr/bin/env python2
import os
import subprocess32
import sys
def recogniser(imagePath):

    directory = os.path.expanduser("~")
    print "changing directory......"
    os.chdir(directory + "/openface")
    print "current directory: %s" % os.getcwd()

    images = []
    for root, dirs, files in os.walk(imagePath):
        images = files
    for image in images:
        command = "./demos/classifier.py infer /home/anonymous/test/features/classifier.pkl {}".format(image)
        cmd = command.split()
        print "\nmaking prediction........\n"
        subprocess32.call(cmd)

if __name__ == '__main__':
    recogniser()
