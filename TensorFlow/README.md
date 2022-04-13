# Introduction to TensorFlow Image Classification

## To start off

Install TensorFlow using `pip install tensorflow`

## What makes TensorFlow a useful tool for learning ML?

- Graphs
- Sessions

Quick explanation of eager-execution and graph execution:
- eager-execution is what most programmers do to execute code, which is just immediate operation evaluations
- TensorFlow builds graphs of operations, and you use a ``Session`` object with passed input and output tensors, that would then compile it with ``session.run()``

Example:
```
@tf.functio
def add(a, b):
  return a + b

add(tf.ones([2, 2]), tf.ones([2, 2]))
```

The `@tf.function` indicates that we want the add function to be converted into a TensorFlow `Function`



# Load the data

```
cifar10 = tf.keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images.shape
```

Let's take a look at some of the images in the dataset-
```
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(20,20))
for i in range(36):
    plt.subplot(6,6,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
```

# Define the model

```
from tensorflow.keras import datasets, layers, models
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()
```


# Compile and fit the model

```
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))
```

With NNs, you see a lot of different optimizers and loss functions, which are pivotal to the functionality of NNs. Test out different optimizers and loss functions to test whether the accuracy improves.

# Visualize the results Epoch by Epoch

```
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
```

```
print(test_acc)
```

How was your accuracy? 

# How can we improve the accuracy?
















Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
