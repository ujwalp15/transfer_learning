<h1 align="center">Transfer learning Inception model</h1>

## Introduction

TensorFlow is an open source library for numerical computation, specializing in machine learning applications. This repository contains 2 scripts,
- retrain.py
- classify.py

What are we going to be building?

We will be using transfer learning, which means we are starting with a model that has been already trained on another problem. We will then be retraining it on a similar problem. Deep learning from scratch can take days, but transfer learning can be done in short order.

We are going to use the Inception v3 network. Inception v3 is trained for the ImageNet Large Visual Recognition Challenge using the data from 2012, and it can differentiate between 1,000 different classes, like Dalmatian or dishwasher. We will use this same network, but retrain it to tell apart a small number of classes based on our own examples.

## Installation

Let us first install tensorflow on to our system. Run the following command in terminal.

```
sudo pip install tensorflow
```

Let us now, check if tensorflow is successfully installed. Run the following code in a python script.

```python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session() # this should print some warnings here, no worries
print(sess.run(hello))
```

If everything is right, then you should see the output as “Hello, TensorFlow!”

## Downloading dataset

We will be using a flower dataset which is readily available online. Let us create a working directory,

```
mkdir transfer_learning && cd transfer_learning
```

Now let us download the flower data set

```
curl -O https://goo.gl/srnxJm
```

once the download is complete, run this command,

```
tar -xzf flower_photos.tgz
```

## Optional, but I have a really basic system

Let's reduce the number of images per category, to speed up the training.

Since all the image file names start with digits, the following command will reduce the number of images by 70%

```
ls flower_photos/roses | wc -l

rm flower_photos/*/[3-9]*

ls flower_photos/roses | wc -l
```

Similarly reduce the number of images of other flowers too.

## Retraining the inception model

Now is the time to fetch the one of the script from my repository. Run the following command in the pwd.

```
curl -O https://raw.githubusercontent.com/ujwalp15/transfer_learning/master/retrain.py
```

We can visualize and monitor the training process in background by using TensorBoard. Before training, start tensorboard by running the following command

```
tensorboard --logdir training_summaries &
```

Now lets start retraining the inception model to start pridiction different types of flowers, run the following command in the terminal,

```
python retrain.py \
--bottleneck_dir=bottlenecks \
--how_many_training_steps=500 \
--model_dir=inception \
--summaries_dir=training_summaries/basic \
--output_graph=retrained_graph.pb \
--output_labels=retrained_labels.txt \
--image_dir=flower_photos
```

Now, I know you guys want to know what these options mean, so here it is,

python retrain.py \ -- Runs the script
--bottleneck_dir=bottlenecks \ -- Bottleneck directory
--how_many_training_steps=500 \ -- Training steps
--model_dir=inception \ -- Path to inception model
--summaries_dir=training_summaries/basic \ -- Path: save summaries
--output_graph=retrained_graph.pb \ -- Path: Retrained graph
--output_labels=retrained_labels.txt \ -- Path: Retrained labels
--image_dir=flower_photos -- Path: Dataset directory

You can google each of them, I will explain the training process in some days...

## Using the newly Trained model

Our model has been trained and the retrained graph can be found in the working directory and is named retrained_labels.pb
The relevant training labels can be found in the working directory and the file is named retrained_labels.txt
We will create a python script to classify images.

Now, we need to classify the images, here we will use the 2nd script from my repository, run the following command in terminal.

```
curl -O https://raw.githubusercontent.com/ujwalp15/transfer_learning/master/classify.py
```

Lets us run the script,

```
python classify.py <image_path>
```

where <image_path> is the path to any image of flower
