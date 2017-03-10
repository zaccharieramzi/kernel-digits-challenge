#!/usr/bin/env python
import os
from sys import exit

from pipeline import main as pipeline

# checking datasets presence
if not(os.path.exists("data/Xtr.csv") and os.path.exists("data/Xte.csv") and
       os.path.exists("data/Ytr.csv")):
    exit("You need to put the files Xtr.csv Xte.csv and Ytr.csv in a folder"
         " named data !")

print("\t\t - Generating bigger images with Matlab - \t\t")

os.chdir("matlab-preprocessing")
matlab = "matlab -nodisplay -nosplash -nodesktop -r \"run('"
os.system(matlab + "zoom_color.m'); exit;\"")
os.system(matlab + "zoom_test_color.m'); exit;\"")
os.chdir("..")

# checking that matlab worked
if not(os.path.exists("data/Xtr63.csv") and os.path.exists("data/Xte63.csv")):
    exit("Matlab script failed, make sure that matlab is installed and in your"
         " path")

print("\t\t - Running the main pipeline - \t\t")
pipeline()
