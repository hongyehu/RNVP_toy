# RNVP_toy
This is a toy model of RNVP flow model, where the neural network is implemented as a fully connected resnet.
It contains the basic ingredients to define a flow model: 1) a prior distribution, defined in Source/source.py, 2) a bijector, defined in layer/flow.py.
By the bijector you defined (here is RNVP), it will "drive" the prior distribution to a complicated target distribution.
![Image of Flow](https://github.com/hongyehu/RNVP_toy/blob/master/demo.jpg)
