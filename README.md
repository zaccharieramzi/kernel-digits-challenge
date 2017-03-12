# Requirements
You need python3.5 to run this project as well as Matlab R2016b.
## Lib installation
All the python libraries you need are listed in `requirements.txt`. To install
them conveniently you can do the following:
```shell
pip install -r requirements.txt
```
We recommend to do that in a virtual environment since some libraries will be
hard to install otherwise. To create a virtual environment, do the following:
```shell
virtualenv -p /usr/bin/python3.5 venv
```
To then activate it you just need to do:
```shell
source venv/bin/actvate
```
## Data
You need to download the data from the Kaggle challenge and put it in a folder
named data at the root of the directory.

# Use
## Easy script
You can run the whole pipeline by calling `python easy_script.py`.
Make sure that **matlab** is in your PATH ! This scripts call the two parts outlined below :
### Big image generation
We used Matlab to generate larger images. To do so run `zoom_color.m` and
`zoom_test_color.m` in the folder *matlab_preprocessing*. It will generate to `.csv`
files that you can load easily with the built functions. 
### Feature computation and classification
To compute HOG features and run the SVM classifier you can run `pipeline.py`

## Notebooks
The main notebooks (*gradient_viz*, *grid_search_color* and *grid_search*) are
relatively easy to use and pretty self forward. To open Jupyter, do the
following:
```shell
jupyter notebook
```

Don't forget to activate tqdm notebook extensions if you want a nice experience.
```shell
jupyter nbextension enable --py widgetsnbextension
```
## Submissions
Your submissions will end up in the submission folder.
