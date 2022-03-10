# *(OPTIONAL BUT RECOMMENDED)*

### Setting up virtual python environment
1. Open a command prompt and run ``python3 -m venv {your_env_name}``
2. On Windows, run ``{your_env_name}\Scripts\activate.bat``
3. On Unix or MacOS, run `` source {your_env_name}/bin/activate``
4. Now you can use ``pip install`` in this v-env local to just this folder

Make sure to install the required modules using ``pip install notebook matplotlib


# What is PyTorch

- PyTorch was developed to make Deep Learning more flexible and easy to work with
- PyTorch revolves around Tensors
- It is intuitive and easy to learn


To install PyTorch:
1. Use ``pip3 install torch torchvision torchaudio``
Import the following modules-
```
%matplotlib inline
import torch
import numpy as np
```

# Tensors

- Tensors are similar to numpy arrays, except Tensors can be used on a GPU to speed up computation
- They are multideminsional matrices
- Used as inputs and outputs of models

We can initialize a tensor like so-
``torch.Tensor(x, y)``, where x and y are dimensions
```
torch.Tensor(5, 3).uniform_(-1, 1)
```

Tensors shape can be defined by tuples, ``(3,4)``, 
```
shape = (3,4)
rand_tensor = torch.rand(shape)
```

Tensors can be created from numpy nd arrays, and vice versa, so they share the same memory location
```
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

print(f"Numpy np_array value: \n {np_array} \n")
print(f"Tensor x_np value: \n {x_np} \n")

np.multiply(np_array, 2, out=np_array)

print(f"Numpy np_array after * 2 operation: \n {np_array} \n")
print(f"Tensor x_np value after modifying numpy array: \n {x_np} \n")
```

The attributes of a Tensor can be described as ``shape``, ``dtype``, ``device``.

```
tensor = torch.rand(4, 5)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```

## Brief explanation of GPUs

- Tensors can be transferred to GPUs from CPUs(tensors are stored here by default)
- GPUs have thousands of cores, while CPUs have up to 16 cores
- CPU cores run in sequential order, while GPU cores run in parallel, meaning tasks are divided and computed among different cores
- PyTorch can use Nvidia CUDA library to use GPU cards

```
if torch.cuda.is_available():
  tensor = tensor.to('cuda')
```

## Indexing and slicing

Similar to numpy array indexing and slicing:
```
tensor = torch.rand(4, 4)
print(tensor)
print('First row: ',tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)
```

# Operations

Adding to all elements in Tensors are simple-
```
print(tensor + 2)
print(tensor)
```
Does not change ``tensor``.

Matrix multiplication can be done multiple ways-
```
x1 = tensor @ tensor.T
print(x1)
x2 = tensor.matmul(tensor.T)
print(x2)
x3 = torch.matmul(tensor, tensor.T, out=y3)
print(x3)
```

# In place operations

- These are operations that store the result in the tensor and change it
- Can be used with the ``_`` suffix

```
print(tensor, "\n")
tensor.add_(5)
print(tensor)
```

# Datasets and Dataloaders

- Two data primitives provided by PyTorch: ``torch.utils.data.DataLoader`` and ``torch.utils.data.Dataset``
- ``Dataset`` stores the samples and labels, while ``Dataloader`` wraps an iterable around the dataset

To start with, import the necessary modules:
```
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
```

//note: if this doesn't work for you, then go on google colab for the rest of the workshop

Pytorch offers simple methods to retrieve training and test data-
```
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
```

We can see what the data looks like just by tinkering with ``training_data``
```
img, label = training_data[13]
plt.title(label)
plt.imshow(img.squeeze(), cmap='gray')
```

Dataloader wraps an iterable through the Dataset and makes machine learning easy to mess with.

```
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```
- This puts all the necessary elements into the Dataloader type
- ``batch_size`` is the amount of data that is processed per batch
- ``shuffle=true`` specifies that the data is reshuffled after every epoch

Visualizing the features and labels-
```
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```
*Note: there are 10 classes for the labels*


# Neural Networks

- A Neural Network, in it simplest term, is a network of neurons connected by layers
- A neuron can be represented as a single computing unit that performs calculations
- There are 3 types of layers: ``input`` layer, ``hidden`` layer, ``output`` layer
- Meant to mimic the way the human brain processes information

``torch.nn`` provides everything regarding neural networks. In fact, every module within PyTorch subclasses ``nn``, so PyTorch's complex structure handles everything in the high level

# Creating the NeuralNetwork class

To start off, we make our neural network class that will inherit from ``nn.module``. the ``__init__`` method will initialize the three layers.

*Note: Every sublcass of ``nn.module`` implements the operations on input data in the ``forward`` method


```
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

```

- The input layer contains 28 × 28 features (from the image dimensions)
- The first ``nn.Linear`` module transforms those 784 features to 512 output features within the hidden layer
- ``nn.ReLu()`` is an activation function that looks something like this ``output = max(0, input)`` meaning it outputs 0 or a positive number. Stands for *Rectified Linear Unit*. If you're interested in this: check this [link](https://www.analyticsvidhya.com/blog/2020/01/fundamentals-deep-learning-activation-functions-when-to-use-them/) out
-  The third ``nn.Linear`` module takes the features from the second hidden layer and transform them into 10 features for the output layer (the number of labels)

Next, we create an instance of the ``NeuralNetwork`` class, and move it to ``device``.
```
model = NeuralNetwork().to(device)
print(model)
```

Pass the input data to the ``model`` we defined. This calls the ``forward`` method.

```
X = torch.rand(1, 28, 28, device=device)
logits = model(X) 
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
```

So we have defined and created our model. The next step is to train it, but there are other stuff we need to handle before that.

# Before we train the model

- The main component to how the neural network determines its accuracy is a loss function
- The loss function calculates the difference between the expected output and the actual output that a neural network produces
- We want this number to be as close to zero as possible
- The most common algorithm used to train these models is called ``back propagation``
- It utilizes the gradient of the loss function to adjust the given parameter
- Back propagation traverses backwards through the network to adjust the parameters(model weights) and retrain the model
- This sounds like it would take a lot of time, but PyTorch has their own built-in function ``torch.autograd`` that automatically calculates the gradient of any computational graph

Before we train our model, there are certain parameters that we must go over-

- Number of Epochs - the number of times the entire training dataset is pass through the network (iterations).
- Batch Size - the number of data samples seen by the model in each epoch. Iterates are the number of batches needs to compete an epoch.
- Learning Rate - the size of steps the model match as it searchs for best weights that will produce a higher model accuracy. Smaller values means the model will take a longer time to find the best weights, while larger values may result in the model step over and misses the best weights which yields unpredictable behavior during training.

```
learning_rate = 1e-3
batch_size = 64
epochs = 5
```

Common loss functions-
1. ``torch.nn.L1Loss``: Mean Absolute Error - *loss(x, y) = |x - y|*
2. ``torch.nn.MSELoss``: Mean Squared Error Loss - *loss(x, y) = (x - y)^2*
3. ``torch.nn.NLLLoss``: Negative Log-Likelihood - *loss(x, y) = - log(y)*
4. ``torch.nn.CrossEntropyLoss``: Cross-Entropy Loss = *loss(x, y) = -Σ(xlog(y))*

We will be using-
```
loss_fn = nn.CrossEntropyLoss()
```

Optimizing Algorithms-
- We need an algorithm that will optimize the model's parameters so it can reduce error in every step
- There are several optimizing algorithms that work for different models, but we will be using Stochastic Gradient Descent(SGD)

```
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

# Training the model

```
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):        

        pred = model(X)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

This is our loop for training a single iteration, or epoch.
``optimizer.zero_grade()`` resets the gradients, since gradients add up by default
``loss.backward()`` is back propagation algorithm
``optimizer.step()`` adjusts the parameters with the gradients we just collected from back propagation

Loop for testing-
```
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

Putting it all together-
```
learning_rate=1e-3
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
```

This will take some time to finish as it's doing a lot of calculations for each epoch.

# Saving models

```
torch.save(model.state_dict(), "data/model.pth")

print("Saved PyTorch Model State to model.pth")
```

Models can be saved with the '.pth' or '.pt' file extension

# Loading models

```
model = NeuralNetwork()
model.load_state_dict(torch.load('data/model.pth'))
model.eval()
```

# Conclusion

Last test to evaulate specific predictions-
```
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
```


# What's next?

- You can test out other datasets and learn how to implement different models. Here's a [link to the datasets PyTorch provides](https://pytorch.org/vision/stable/datasets.html)
- I recommend look at the MNIST Handwritten-Digit Recognition dataset. It is a great beginner project, and you can add on to it very easily. For example, [this link uses the MNIST dataset, without PyTorch, but you can implement it using PyTorch to demonstrate your knowledge of it](https://data-flair.training/blogs/python-deep-learning-project-handwritten-digit-recognition/)
- Look at PyTorch documentation if you really want to understand the functions it includes ([PyTorchTutorials](https://pytorch.org/tutorials/))
