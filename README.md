# Multipale Image Neural Representations
Based on 
[Implicit Neural Representations with Periodic Activation Functions](https://github.com/vsitzmann/siren)

## Setup enviroment

You can then set up a conda environment with all dependencies like so:
```
conda env create -f environment.yml
conda activate ImageRep
```
## Train image representation
run training
The image experiment can be reproduced with
```
python experiment_scripts/train_img.py --model_type=sine
```
please make sure to login to [wandb](https://docs.wandb.ai/quickstart) before staring training
the checkpoints stored at "output" folder.
### Evaluate
to produce evaluation images
```
python eval_trained.py [checkpoint path]
```
the result will be two image
####Size Interpolation
![plot](./size_interpolation_example.jpg)

####Image Index Interpolation
![plot](./image_interpolation_example.jpg)


