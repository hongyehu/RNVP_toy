# RNVP_toy
This is a toy model of RNVP flow model, where the neural network is implemented as a fully connected resnet.
It contains the basic ingredients to define a flow model: 
* 1) a prior distribution, defined in ```Source/source.py```
* 2) a bijector, defined in ```layer/flow.py```.
By the bijector you defined (here is RNVP), it will "drive" the prior distribution to a complicated target distribution.

In the following picture, the left figure is latent distribution, which is Laplace distribution here. The right figure is the target distribution, which is pinwheel having four tentacles. And the middle figure is the distribution transformed from prior distribution by the bijector, which is RNVP here.
![Image of Flow](https://github.com/hongyehu/RNVP_toy/blob/master/demo.jpg)
