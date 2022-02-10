# Welcome to the first SIG AI workshop of the semester!
### Here, we'll go over a quick tutorial on setting up Jupyter notebook and working with a dataset.

## Anaconda

1. First, check if you've already installed Anaconda by opening a command prompt and typing conda
2. If you don't have it installed, then go to [Anaconda installation](https://www.anaconda.com/products/individual) to install anaconda based on your OS
3. If you have python installed, but not through Anaconda, then open a command prompt and type **pip install jupyter**
4. Once you've got jupyter installed, we're also going to need other modules
5. Anaconda comes with most of the modules needed for machine learning, which are *pandas*, *matplotlib*, *numpy*, and *scikit-learn*.
6. Using pip install, we can install these modules by running **pip install pandas matplotlib numpy scikit-learn**
7. Now, launch jupyter notebook by going to anaconda navigator and launch jupyter notebook, or typing **jupyter notebook** in command prompt

#### Anaconda navigator should look similar to this-
<img width="1422" alt="anaconda_example" src="https://user-images.githubusercontent.com/61035833/153460548-46b5fb98-51ae-4c29-9a06-6c5e87d6151e.png">


make sure to click on notebook, and not on jupyterlab

## Jupyter start-up

#### The start-up displays all your user folders
<img width="1429" alt="jupyter_start" src="https://user-images.githubusercontent.com/61035833/153461781-8b54a444-e984-4000-8f06-960242cf64a1.png">

- Best practice is to create a new folder in which you'll have your files in, so click on *new* and create folder
- Next, create new python file
- Now, you have your notebook set up and ready to work with!


## Importing the required modules

Copy and paste this code snippet into your first cell and run it
```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
```
## Using pandas to read datasets

This code snippet allows the user to read a dataset from the UCI Machine Learning repository and store as a DataFrame
```
iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
```

