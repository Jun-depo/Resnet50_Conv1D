# Resnet50_Conv1D

The code was written to solve wave data or other sequence model problems. Residual neural network was one of major convoltional network (CNN) advances by [He et. al. 2015](https://arxiv.org/abs/1512.03385). The initial work used two dimensional convolution (conv2d) for image analyses. I modified it to conv1d to analyze sequence data for a [Kaggle LANL-Earthquake-Prediction competition](https://www.kaggle.com/c/LANL-Earthquake-Prediction).  I used the code from Resnet_conv2d coding assigment of [Andrew Ng's CNN class](https://www.coursera.org/learn/convolutional-neural-networks) as the starting reference using Keras library. I add a Jupyter notebook example using sin(x) synthetic data.  

Files:
* Resnet50Conv1d_BN.py 
* Resnet50Conv1d_sin_BN.ipynb

The code was written for the regression. Need to uncomment some code at the end to use "sigmoid" or "softmax" activation for classification tasks.  

### License:
The MIT License. 
