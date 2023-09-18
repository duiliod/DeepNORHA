# # PyTorch Implementation of DeepNORAH
This repository provides a [PyTorch](https://pytorch.org/) implementation of the *DeepNORAH* method presented in our MICCAI 2023 paper ”Learning normal asymmetry representations for homologous brain structures”.

## Citation and Contact
You find a PDF of the Learning normal asymmetry representations for homologous brain structures MICCAI 2023 paper at 
[https://ignaciorlando.github.io/publication/2023-miccai/](https://ignaciorlando.github.io/publication/2023-miccai/).

If you use our work, please also cite the paper:
```
@inproceedings{deangeli2023learning,
  title={Learning normal asymmetry representations for homologous brain structures},
  author={Deangeli, Duilio and Iarussi, Emmanuel and Princich, Juan Pablo and Bendersky, Mariana and Larrabide, Ignacio and Orlando, José Ignacio},
  booktitle={26th International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year={2023},
  publisher={Springer}
}
```

If you would like to get in touch, please contact [ddeangeli@pladema.exa.unicen.edu.ar](ddeangeli@pladema.exa.unicen.edu.ar).

## Abstract
> > Although normal homologous brain structures are approximately symmetrical by definition, they also have shape differences
> > due to e.g. natural ageing. On the other hand, neurodegenerative conditions induce their own changes in this asymmetry, 
> > making them more pronounced or altering their location. Identifying when these alterations are due to a pathological 
> > deterioration is still challenging. Current clinical tools rely either on subjective evaluations, basic volume measurements
> > or disease-specific deep learning models. This paper introduces a novel method to learn normal asymmetry patterns in 
> > homologous brain structures based on anomaly detection and representation learning. Our framework uses a Siamese architecture 
> > to map 3D segmentations of left and right hemispherical sides of a brain structure to a normal asymmetry embedding space, 
> > learned using a support vector data description objective. Being trained using healthy samples only, it can quantify
> > deviations-from-normal-asymmetry patterns in unseen samples by measuring the distance of their embeddings to the center of 
> > the learned normal space. We demonstrate in public and in-house sets that our method can accurately characterize normal 
> > asymmetries and detect pathological alterations due to Alzheimer’s disease and hippocampal sclerosis, 
> > even though no diseased cases were accessed for training.

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

## Training example
```
cd <path-to-DeepNORAH-directory>
# activate virtual environment
source myenv/bin/activate  # or 'source activate myenv' for conda
# change to source directory
cd src/trainer
# run experiment
python train_SVDD.py --configAE '../../configs/AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filters.yml' --configSVDD '../../configs/AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filtersSVDD.yml'
```
This example trains a DeepNORAH model where healthy hippocampi is considered to be the normal class. 

## License
MIT
