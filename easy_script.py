#!/usr/bin/env python
import os

from pipeline import main as pipeline

print("\t\t - Generating bigger images with Matlab - \t\t")

os.chdir("matlab-preprocessing")
matlab = "matlab -nodisplay -nosplash -nodesktop -r \"run('"
os.system(matlab + "zoom_color.m'); exit;\"")
os.system(matlab + "zoom_test_color.m'); exit;\"")

print("\t\t - Running the main pipeline - \t\t")
os.chdir("..")
pipeline()
