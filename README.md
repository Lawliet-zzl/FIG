# Code for "Revealing Distributional Vulnerability of Explicit Discriminators by Implicit Generators"

## requirement
* Python 3.7
* Pytorch 1.1
* scikit-learn
* tqdm
* pandas
* scipy

## Datasets
### In-distribution Datasets
* [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
* [SVHN](http://ufldl.stanford.edu/housenumbers/)

Our codes will download the two in-distribution datasets automatically.

### Out-of-Distribtion Datasets
The following four out-of-distribution datasets are provided by [ODIN](https://github.com/ShiyuLiang/odin-pytorch)
* [TinyImageNet (r)](https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz)
* [TinyImageNet (c)](https://www.dropbox.com/s/avgm2u562itwpkl/Imagenet.tar.gz)

Each out-of-distribution dataset should be put in the corresponding subdir in [./data_FIG](./data_FIG)

## Train and Test
Run the script [demo.sh](./code_FIG/demo.sh). 
