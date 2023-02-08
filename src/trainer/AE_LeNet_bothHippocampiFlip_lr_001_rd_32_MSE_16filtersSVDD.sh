#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --job-name="AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filtersSVDD"
#SBATCH --output=AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filtersSVDD.out
#SBATCH --gres=gpu:1
#SBATCH --qos=longrunning

python train_SVDD.py --configAE '/home/ddeangeli/deepsvdd/configs/AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filters.yml' --configSVDD '/home/ddeangeli/deepsvdd/configs/AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filtersSVDD.yml'
python train_SVDD.py --no-pretrained --configAE '/home/ddeangeli/deepsvdd/configs/AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filters.yml' --configSVDD '/home/ddeangeli/deepsvdd/configs/AE_LeNet_bothHippocampiFlip_lr_001_rd_32_MSE_16filtersSVDD.yml'

