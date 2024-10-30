# Effective Combinations and Variations of Pretext Tasks for Downstream Tasks in Classification

## Research Question
* How do different combinations and sequences of pretext tasks impact the performance of downstream classification tasks?
  * Traditional single pretext task vs multi pretext task
  * Simultaneous vs curriculum multi pretext tasks
  * Identifying effective combinations of pretext tasks to enhance classification performance.

## Literature review
[Mixture of Self-Supervised Learning](https://arxiv.org/abs/2307.14897)					
Ruslim, A.R., Yudistira, N., & Setiawan, B.D. (2023)

[Improved skin lesion recognition by Self-Supervised Curricular Deep Learning approach](https://arxiv.org/pdf/2112.12086) 						
Sirotkin, Escudero-Vinolo, et al. (2021)

[A Novel Multi-Task Self-Supervised Representation Learning Paradigm](https://ieeexplore.ieee.org/document/9456562)
Li et al., IEEE AIID 2021

[Weak Augmentation Guided Relational Self-Supervised Learning](https://ieeexplore.ieee.org/abstract/document/10540667?casa_token=Z6tggRpPZdYAAAAA:CBlbpl-pyi9ZOouDDsQ3TgFIL_1c55-Jy7iB1kFu1Hr7-YhwskbgsW_h9jM3aEOD_bJNSGs_Cls)
Zheng, Mingkai, et al. (2022)

[Colorful Image Colorization](https://arxiv.org/pdf/1603.08511)
Richard Zhang, Phillip Isola, Alexei A. Efros (2016)

[Context Encoders: Feature Learning by Inpainting](https://arxiv.org/pdf/1604.07379)
Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, Alexei A. Efros (2016)

## Model Architecture
We utilized a ResNet-18 architecture for our experiments. ResNet-18 is a convolutional neural network that is 18 layers deep, known for its ability to handle vanishing gradient problems through residual learning.

## Datasets
* [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
* [STL10](https://cs.stanford.edu/~acoates/stl10/)

## Pretext Tasks
1. Rotation
2. Colorization
3. Inpainting
4. Jigsaw
