# Transfer learnings of the Inception model

## Introduction

This tutorial aims at retraining a model to recognise images of flowers. We will retrain the last layer of Google's Inception-v3 model. This will be done using the science of transfer learning.

Below you will find some notes on some of the topics mentioned above.

### TensorFlow

TensorFlow is an open source library for numerical computation, specializing in machine learning applications.

### Transfer learning

Transfer learning allows us to leverage the knowledge gained while solving one problem and applying it to a different but related problem.

### An introduction to Inception-v3

Inception v3 is the 2015 iteration of Google's Inception architecture for image recognition. Inception-v3 is trained for the ImageNet Large Visual Recognition Challenge using the data from 2012\. This is a standard task in computer vision, where models try to classify entire images into 1000 different classes, like "Zebra", "Dalmatian", and "Dishwasher".

### Why use transfer learning?

Before I answer the question, I would like to talk about what a pre-trained model is: Simply put, a pre-trained model is a model created by someone else, trying to solve a similar problem.

For example, if we want to recognise the image of a flower we can spend years at developing a model for it or we can leverage the knowledge of the Inception model to identify those images.

As, mentioned above Inception is a pre-trained image classifier; it would serve our purpose if we mould it to our needs.

--------------------------------------------------------------------------------

**_Note: This tutorial assumes that, one is akin to the concepts of python programming, and that one knows the basics of Linux operating system._**

--------------------------------------------------------------------------------

## Building our model

We will be using transfer learning, to retrain the inception network to tell apart a few different classes of flowers

This repository contains the following two scripts:

- retrain.py
- classify.py

### Step 1: Installation

Let us first install TensorFlow on our system. Run the following command in terminal to do so.

```sh
$sudo pip install tensorflow
```

To check if TensorFlow has been successfully installed, run the following code as a python script.

```python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session() # this should print some warnings here, no worries
print(sess.run(hello))
```

If all went well, your output will be:

`Hello, TensorFlow!`

### Step 2: Downloading dataset

We will be using a flower dataset which is readily available online. Let us create a working directory.

Do the following on your shell:

```sh
$mkdir transfer_learning && cd transfer_learning
```

Now let us download the flower data set.

Run the following command:

```sh
$curl -O https://goo.gl/srnxJm
```

once the download is complete, run this command:

```sh
$tar -xzf flower_photos.tgz
```

#### Reduce the size of the data

_Optional, but it would be useful people who have a basic system. This step speeds up the training._

Since all the image file names start with digits, the following command will reduce the number of images by 70%

```sh
$ls flower_photos/roses | wc -l
$rm flower_photos/*/[3-9]*
$ls flower_photos/roses | wc -l
```

Similarly reduce the number of images of other flowers types too.

### Step 3: Retraining the inception model

Now is the time to fetch the one of the script from my repository. Run the following command in the PWD:

```sh
$curl -O https://raw.githubusercontent.com/ujwalp15/transfer_learning/master/retrain.py
```

We can visualise and monitor the training process in the background by using TensorBoard. Before training, start TensorBoard by running the following command:

```sh
$tensorboard --logdir training_summaries &
```

Now lets start retraining the inception model to start prediction different types of flowers, run the following command in the terminal:

```sh
$python retrain.py \ # Runs the script
--bottleneck_dir=bottlenecks \ # Bottleneck directory
--how_many_training_steps=500 \ # Training steps
--model_dir=inception \ # Path to inception model
--summaries_dir=training_summaries/basic \ # Path: save summaries
--output_graph=retrained_graph.pb \ # Path: Retrained graph
--output_labels=retrained_labels.txt \ # Path: Retrained labels
--image_dir=flower_photos # Path: Dataset directory
```

You can google each of them, I will explain the training process in some days...

### Step 4: Using the newly Trained model

Our model has been trained and the retrained graph can be found in the working directory: a file named "retrained_labels.pb".

The relevant training labels can also be found in the working directory: a file named "retrained_labels.txt".

We will now create a python script to classify images.

We will need to use the second script form my repository.

```sh
$curl -O https://raw.githubusercontent.com/ujwalp15/transfer_learning/master/classify.py
```

Lets us run the script,

`$python classify.py target_image`

where, target_image is the path to any image of flower you would like to label.
