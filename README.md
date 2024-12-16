# Effective Combinations and Variations of Pretext Tasks for Downstream Tasks in Classification

## Research Question
* How do different combinations and sequences of pretext tasks impact the performance of downstream classification tasks?
  * Traditional single pretext task vs multi pretext task
  * Simultaneous vs curriculum multi pretext tasks
  * Identifying effective combinations of pretext tasks to enhance classification performance.

## References

- **Colorful Image Colorization**  
  Richard Zhang, Phillip Isola, Alexei A. Efros. *arXiv preprint, 2016*.  
  [arXiv:1603.08511](https://arxiv.org/abs/1603.08511)

- **Context Encoders: Feature Learning by Inpainting**  
  Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, Alexei A. Efros. *arXiv preprint, 2016*.  
  [arXiv:1604.07379](https://arxiv.org/abs/1604.07379)

- **Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles**  
  Mehdi Noroozi, Paolo Favaro. *arXiv preprint, 2017*.  
  [arXiv:1603.09246](https://arxiv.org/abs/1603.09246)

- **Self-Supervised Learning — Lightly 1.5.15 Documentation**  
  Lightly.ai. *Documentation, 2022*.  
  [Lightly.ai Documentation](https://docs.lightly.ai/self-supervised-learning/getting_started/lightly_at_a_glance.html)

- **Multi-Task Learning in ML: Optimization & Use Cases [Overview]**  
  Kundu, R. *V7labs Blog, 2022*.  
  [V7labs](https://www.v7labs.com/blog/multi-task-learning-guide#loss-construction)

- **Self-Supervised Representation Learning by Rotation Feature Decoupling**  
  Zeyu Feng, Chang Xu, Dacheng Tao. *2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.  
  [DOI:10.1109/CVPR.2019.01061](https://doi.org/10.1109/CVPR.2019.01061)

- **Exploring Simple Siamese Representation Learning**  
  Xinlei Chen, Kaiming He. *2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.  
  [DOI:10.1109/CVPR46437.2021.01549](https://doi.org/10.1109/CVPR46437.2021.01549)

- **Momentum Contrast for Unsupervised Visual Representation Learning**  
  Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick. *arXiv preprint, 2020*.  
  [arXiv:1911.05722](https://arxiv.org/abs/1911.05722)

- **Improved Skin Lesion Recognition by a Self-Supervised Curricular Deep Learning Approach**  
  Kirill Sirotkin, Marcos Escudero-Viñolo, Pablo Carballeira, Juan Carlos SanMiguel. *arXiv preprint, 2021*.  
  [arXiv:2112.12086](https://arxiv.org/abs/2112.12086)

- **A Novel Multi-Task Self-Supervised Representation Learning Paradigm**  
  Yinggang Li, Junwei Hu, Jifeng Sun, Shuai Zhao, Qi Zhang, Yibin Lin. *2021 IEEE International Conference on Artificial Intelligence and Industrial Design (AIID)*.  
  [DOI:10.1109/AIID51893.2021.9456562](https://doi.org/10.1109/AIID51893.2021.9456562)

- **Unsupervised Representation Learning by Predicting Image Rotations**  
  Spyros Gidaris, Praveer Singh, Nikos Komodakis. *arXiv preprint, 2018*.  
  [arXiv:1803.07728](https://arxiv.org/abs/1803.07728)


## Model Architecture
We utilized a ResNet-18 architecture for our experiments. ResNet-18 is a convolutional neural network that is 18 layers deep, known for its ability to handle vanishing gradient problems through residual learning.

## Datasets
* [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
* [STL10](https://cs.stanford.edu/~acoates/stl10/)
* [CALTECH101](https://data.caltech.edu/records/mzrjq-6wc02)

## Pretext Tasks
1. Rotation
2. Colorization
3. Inpainting
4. Jigsaw
5. SimSiam
6. MoCo
