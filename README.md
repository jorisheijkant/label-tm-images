# Label images with a Teachable Machine model

In this repository, some sample code is written to analyze a folder of images with a [Teachable Machine](https://teachablemachine.withgoogle.com/) trained model. It outputs a sorted folder of images and a spreadsheet. The code has been created as part of a course nin machine learning.

## For those of you who do not know git

There's two ways to use this code on your machine. Simply download the code and navigate with your terminal to the folder where you downloaded it. Follow the steps below to run the code. Better even is to use [git](https://git-scm.com/) and the `git clone` command to fetch the code. Or to create a fork of this repo to modify it to your own needs.

## Other prerequisites

The code is written in Python. Preferably use a Python virtual environment manager such as Conda or venv to create a seperate environment to run the code. I've succesfully used Python 3.12 to run this.

Before running the script, install the needed libraries with `pip install -r requirements.txt` or the package manager of your virtual environment.

## Running the script

You need two things before running the script. A model and a folder of images to label. When you trained the model in Teachable Machine, open the export dialog and select the Tensorflow (Keras) option. If you download the file, you should get a (zipped) folder containing a `.h5` file and a `labels.txt` file. Drag these into the `models/` folder of this repository.

Afterwards, select a bunch of images to label using your model. Drag these into the `images/` folder. I would advise to start with just a couple of images to check whether the script is working.

Open the `label_images.py` script in your favorite code editor and rename the variables `model_path` and `label_path` to the correct name of your model. If you did not rename your files they should already be named correctly though. Just make sure you check this.

Then you can run the `python label_images.py` command from the terminal. This should output some code about your progress. After the labeling process is finished, you should have one folder per category in the `output/` folder, and a csv file with all labels.

### N.B.

- The output from Teachable Machine might be created with a different version of Keras. That's why a tool is included in the `utils/` folder to make sure this does not lead to errors when using the more modern version you'll install here.
