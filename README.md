# Deep neural network for image binary classification

### Intro
------------------------------------------------------------------
This is a project following the instruction from a deep learning online course Neural Networks and Deep Learning by Andrew Ng. We built a L layer ANN framework from scratch by implementing the forward and backward propagation as well as supportive functionalities such as image resize and crop, L2 normalization. 

### Datasets
------------------------------------------------------------------
The framework was applied to train two models for two different datasets:

1. An images dataset consist of different kinds of street foods were labeled as "Hotdog" or "Not Hotdog". This dataset was inspired by the famous TV show Silicon Valley. [1] (pending)

2. The original "Cat or Not Cat" dataset used in Andrew Ng's class. 

### Models
----------------------------------------------------------------------
So far totally 3 examples trained for the cat/notcat dataset.

1. A feed forward NN built from scratch. (accuracy 0.8)

2. A feed forward NN built with pytorch. (accuracy 0.8)

3. A CNN model with batch norm, early stop and augmentation implemented with pytorch. (accuracy 0.9)

Reference
--------------------------------------------------
[1] https://www.kaggle.com/dansbecker/hot-dog-not-hot-dog