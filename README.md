# Conditional-Gans
The test code for Conditional Generative Adversarial Nets using tensorflow.

##INTRODUCTION

Tensorflow implements of [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784).The paper should be the first one to introduce Conditional GANS.But they did not provide source codes.There is some difference comparing the paper.The Gans is based on Convolution network and the code refer to [DCGAN](https://github.com/carpedm20/DCGAN-tensorflow).

##Usage

  Download mnist:
  
    $ python download.py
  
  Train:
  
    $ python main.py --operation 0
  
  Test:
  
    $ python main.py --operation 1
  
  Visualization:
  
    $ python main.py --operation 2
    
  GIF:
  
    $ python make_gif.py
  
##Result on mnist

![](images/result.gif)


##Visualization:


the visualization of weights:

![](images/activations.png)

the visualization of activation:

![](images/weights.png)


##Reference code

[DCGAN](https://github.com/carpedm20/DCGAN-tensorflow)
