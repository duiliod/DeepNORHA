# # PyTorch Implementation of DeepNORAH
This repository provides a [PyTorch](https://pytorch.org/) implementation of the *DeepNORAH* method presented in our.......

If you would like to get in touch, please contact [ddeangeli@pladema.exa.unicen.edu.ar](ddeangeli@pladema.exa.unicen.edu.ar).

## Abstract
> > This paper introduce a novel method to learn normal asymmetry patterns in homologous brain structures based on anomaly 
> > detection and representation learning. Current clinical tools rely either on subjective evaluations, basic volume 
> > measurements or disease-specific deep learning models. Instead, our framework uses a Siamese architecture to map 3D 
> > segmentations of left and right hemispherical sides of a brain structure to a normal asymmetry embedding space, learned 
> > using a support vector data description objective. Being trained using healthy samples only, it can quantify 
> > deviations-from-normal-asymmetry patterns in unseen samples by measuring the distance of their embeddings to the center
> > of the normal space. We demonstrate in public and in-house sets that our method can accurately characterize normal 
> > asymmetries and detect pathological alterations due to Alzheimer's disease and hippocampal sclerosis, even though 
> > no diseased cases were accessed for training.

## Installation
This code is written in `Python 3.7` and requires the packages listed in `requirements.txt`.

Clone the repository to your local machine and directory of choice:
```
git clone https://github.com/duiliod/DeepNORHA.git
```

To run the code, we recommend setting up a virtual environment, e.g. using `virtualenv` or `conda`:

### `virtualenv`
```
# pip install virtualenv
cd <path-to-DeepNORAH-directory>
virtualenv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### `conda`
```
cd <path-to-DeepNORAH-directory>
conda create --name myenv
source activate myenv
while read requirement; do conda install -n myenv --yes $requirement; done < requirements.txt
```

## Running experiments

We currently have implemented the MNIST ([http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)) and 
CIFAR-10 ([https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)) datasets and 
simple LeNet-type networks.

Have a look into `main.py` for all possible arguments and options.

### Training example
```
cd <path-to-DeepNORAH-directory>
# activate virtual environment
source myenv/bin/activate  # or 'source activate myenv' for conda
# change to source directory
cd src/trainer
# run experiment
python train_SVDD.py --configAE '/home/ddeangeli/deepsvdd/configs/AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filters.yml' --configSVDD '/home/ddeangeli/deepsvdd/configs/AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filtersSVDD.yml'
```
This example trains a DeepNORAH model where healthy hippocampi is considered to be the normal class. 

## License
MIT
