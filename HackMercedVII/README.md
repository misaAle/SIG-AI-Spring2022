# Welcome to HackMerced VII Intro to ML workshop!
### Here, we'll go over a quick tutorial on setting up Jupyter notebook and going through the stages of project building.

## Anaconda

1. First, check if you've already installed Anaconda by opening a command prompt and typing **conda**
2. If you don't have it installed, then go to [Anaconda installation](https://www.anaconda.com/products/individual) to install anaconda based on your OS 
3. If you have python installed, but not through Anaconda, then open a command prompt and type **pip install jupyter**
4. Once you've got jupyter installed, we're also going to need other modules
5. Anaconda comes with most of the modules needed for machine learning, which are *pandas*, *matplotlib*, *numpy*, and *scikit-learn*.
6. Using pip install, we can install these modules by running **pip install pandas matplotlib numpy scikit-learn**
7. Now, launch jupyter notebook by going to anaconda navigator and launch jupyter notebook, or typing **jupyter notebook** in command prompt

If you're hesitant on installing Anaconda, or installing Python, then I suggest following along with the workshop using Google Colab [Colab](https://research.google.com/colaboratory/) 

#### Anaconda navigator should look similar to this-
<img width="1422" alt="anaconda_example" src="https://user-images.githubusercontent.com/61035833/153460548-46b5fb98-51ae-4c29-9a06-6c5e87d6151e.png">


make sure to click on notebook, and not on jupyterlab

## Jupyter Start-up

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
You now have access to the iris dataset!

We now want to modify the data so it can be more readable and understandable

```
iris.columns = ["sepal length", "sepal width", "petal length", "petal width", "class"]
iris.columns
```
This classifies the columns, or *features*, of the data. The second line also displays the columns.

## Extracting useful information

We want to extract information from the data that will allow us to use it effectively. For example, this dataset has four columns of numerical values, and one column classifying each entry. Meaning, the numerical values are the data we should be handling the most. Using *iloc*, we can extract these four columns and storing it as a numpy array.

```
inputs = iris.iloc[:,0:4].values
```

## Class labeling

We've extracted the numerical values from the original data, but we still want to process our *"class"* column into something we can use.

```
from sklearn import preprocessing
label = preprocessing.LabelEncoder()
targets = label.fit_transform(iris["class"])
```

*'LabelEncoder()'* essentially takes in the *"class"* column and transforms it into numerical labels (0, 1, 2).
- **For reference, 0 = Setosa, 1 = Versicolour, 2 = Viginica**

## Visualizing the data

The data has been processed and is ready to visualize. The best way to visualize data is too look at the data in small portions.

```
plt.scatter(iris["petal length"], iris["petal width"], c = targets)
```

Try changing the columns to visualize the data more, such as *"sepal length"* and *"sepal width"*!

Another cool thing you can do is get a scatter plot of every feature by using the following-
```
pd.plotting.scatter_matrix(iris.iloc[:,0:4], c = targets, figsize = (10,10))
```
This can allow us to look plots from different angles

# Testing out algorithms

- The last thing to do is apply algorithms that will classify and predict the data
- We can test out our own simply algorithm or already implemented algorithms

## Basic Decision Tree

This code snippet is an algorithm I came up with based on the first plot
```
def decision_tree(row):
    if row["petal length"] < 2.5:
        return 0
    elif row["petal length"] < 5.0:
        return 1
    else:
        return 2
    
num_correct = 0
for i, row in iris.iloc[:,0:4].iterrows():
    prediction = decision_tree(row)
    if prediction == targets[i]:
        num_correct +=1
        
print("accuracy =", num_correct/len(iris))
```
As we can see, this algorithm can yield pretty good results despite being simple.
Can you come up with a better decision tree with a better accuracy score?

## KNN Algorithm

Sklearn provides the K Nearest Neighbors algorithm and we can apply it to pretty much any problem
The following snippet imports the algorithm and extracts the necessary data into our variables *X_train, X_test, y_train, y_test*

```
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs, targets)
```

Now we can fit the model using the following-
```
iris_knn = knn.fit(X_train, y_train)
iris_knn.score(X_test, y_test)
```

## Sklearn Decision Tree Algorithm
Sklearn has their own decision tree algorithm that makes these cut-off segments.
```
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
clf.score(X_test, y_test)
```

We can visualize the tree using the following-
```
plt.figure(figsize=(12,12))
tree.plot_tree(clf, fontsize=10)
tree.plot_tree(clf);
```

# Bonus

```
from sklearn.datasets import load_iris

iris_copy = load_iris()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# Parameters
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02


for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
    # We only take the two corresponding features
    X = iris_copy.data[:, pair]
    y = iris_copy.target

    # Train
    clf_copy = DecisionTreeClassifier().fit(X, y)

    # Plot the decision boundary
    plt.subplot(2, 3, pairidx + 1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step)
    )
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = clf_copy.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    plt.xlabel(iris_copy.feature_names[pair[0]])
    plt.ylabel(iris_copy.feature_names[pair[1]])

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(
            X[idx, 0],
            X[idx, 1],
            c=color,
            label=iris_copy.target_names[i],
            cmap=plt.cm.RdYlBu,
            edgecolor="black",
            s=15,
        )

plt.suptitle("Decision surface of decision trees trained on pairs of features")
plt.legend(loc="lower right", borderpad=0, handletextpad=0)
_ = plt.axis("tight")

```

# OpenCV Basics

How to use OpenCV?
```
import cv2 as cv
```

- Imread (Stores an image as a Mat object for images using the path to Image as a string)
- Imshow (Displays the image)
- Imwrite (Saves the image

There are multiple ways to retrieve the path to an image, such as

Most conventional way that is interactive:
```
import easygui
path=easygui.fileopenbox() //returns a string

img = cv.imread(pathath) // saves image to 'img'
```

If you're just dealing with one file, then this is more straightforward:
```
img = cv.imread(cv.samples.findFile("starry_night.jpg"))
```


Showing the image is also pretty simple:
```
cv.imshow("Display window", img)
```

Saving the image using 'Imwrite':
```
cv.imwrite("starry_night.png", img)
```

'Imwrite' is a little more complicated to use as it requires a full path, so you can import OS module to tinker around with folders and files:
```
import os
path1 = os.path.dirname(ImagePath)
extension=os.path.splitext(ImagePath)[1]
path = os.path.join(path1, newName+extension)
```
