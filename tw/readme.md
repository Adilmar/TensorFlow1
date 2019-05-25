# Text Classification with Keras and TensorFlow
## Blog post is [here](https://vgpena.github.io/classifying-tweets-with-keras-and-tensorflow/)

If you want an intro to neural nets and the "long version" of what this is and what it does, read my [blog post](https://vgpena.github.io/classifying-tweets-with-keras-and-tensorflow/).

Data can be downloaded [here](http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/). Many thanks to ThinkNook for putting such a great resource out there.

## Installation

You need Python 2 to run this project; I also recommend [Virtualenv](https://virtualenv.pypa.io/en/stable/) and [iPython](https://ipython.org/).

Run `pip install` to install everything listed in `requirements.txt`.

## Usage
You need to train your net once, and then you can load those settings and use it whenever you want without having to retrain it.

### Training
Change line 10 of `makeModel.py` to point to wherever you downloaded your data as a CSV.

Then run `Python makeModel.py` (or, if you're in iPython, `run makeModel.py`). Then go do something else for the 40-60 minutes that it takes to train your neural net.

When creating the net finishes, three new files should have been created: `dictionary.json`, `model.json`, and `model.h5`. You will need these to use the net.

### Classification
To use the net to classify data, run `loadModel.py` and type into the console when prompted. Hitting Enter without typing anything will quit the program.