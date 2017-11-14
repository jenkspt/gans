
# Improved Training of Wassserstein Gans Notes


# Progressive Growing of Gans for Improved Quality, Stability, and Variation Notes


#### Initialization
*Biases*
All bias parameters initialized to zero
*Weights*
Weights are initialized with unit variance, and then scaled at runtime by the per-layer
normalization constant from He's initializer. This is because of some tricky shit with Adam
and RMSProp optimizers
* [He's paper](https://arxiv.org/pdf/1502.01852.pdf)
* [Code from wgan-gp repo](https://github.com/igul222/improved_wgan_training/blob/fa66c574a54c4916d27c55441d33753dcc78f6bc/tflib/ops/conv2d.py#L69)
* [Code from nvidia repo]()
  * weights are scaled with a layer-specific constant at runtime

#### Mini-Batch Standard Deviation
* We inject the across-minibatch standard deviation as an additional 
feature map at 4×4 resolution toward the end of the discriminator as 
described in Section 3

#### Normalization
* Pixel-wise normalization: see section 4.2
