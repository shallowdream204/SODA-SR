# SODA-SR (CVPR2024)

<a href='https://shallowdream204.github.io/soda-sr-project/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='https://arxiv.org/pdf/2303.17783'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

This repository is an official PyTorch implementation of the paper [Uncertainty-Aware Source-Free Adaptive Image Super-Resolution with Wavelet Augmentation Transformer](https://openaccess.thecvf.com/content/CVPR2024/papers/Ai_Uncertainty-Aware_Source-Free_Adaptive_Image_Super-Resolution_with_Wavelet_Augmentation_Transformer_CVPR_2024_paper.pdf).

[Yuang Ai](https://scholar.google.com/citations?user=2Qp7Y5kAAAAJ)<sup>1,2</sup>, [Xiaoqiang Zhou](https://scholar.google.com/citations?user=Z2BTkNIAAAAJ)<sup>1,3</sup>, [Huaibo Huang](https://scholar.google.com/citations?user=XMvLciUAAAAJ)<sup>1,2</sup>, [Lei Zhang](https://scholar.google.com/citations?user=tAK5l1IAAAAJ)<sup>4,5</sup>, [Ran He](https://scholar.google.com/citations?user=ayrg9AUAAAAJ)<sup>1,2,6</sup>

<sup>1</sup>MAIS & CRIPAC, Institute of Automation, Chinese Academy of Sciences<br><sup>2</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences<br><sup>3</sup>University of Science and Technology of China <sup>4</sup>OPPO Research Institute<br><sup>5</sup>The Hong Kong Polytechnic University <sup>6</sup>ShanghaiTech University


## Abstract
Unsupervised Domain Adaptation (UDA) can effectively address domain gap issues in real-world image SuperResolution (SR) by accessing both the source and target data. Considering privacy policies or transmission restrictions of source data in practical scenarios, we propose a SOurce-free Domain Adaptation framework for image SR (SODA-SR) to address this issue, i.e., adapt a source-trained model to a target domain with only unlabeled target data. SODA-SR leverages the source-trained model to generate refined pseudo-labels for teacher-student learning. To better utilize pseudo-labels, we propose a novel wavelet-based augmentation method, named Wavelet Augmentation Transformer (WAT), which can be flexibly incorporated with existing networks, to implicitly produce useful augmented data. WAT learns low-frequency information of varying levels across diverse samples, which is aggregated efficiently via deformable attention. Furthermore, an uncertainty-aware self-training mechanism is proposed to improve the accuracy of pseudo-labels, with inaccurate predictions being rectified by uncertainty estimation. To acquire better SR results and avoid overfitting pseudo-labels, several regularization losses are proposed to constrain target LR and SR images in the frequency domain. Experiments show that without accessing source data, SODA-SR outperforms state-of-the-art UDA methods in both synthetic→real and real→real adaptation settings, and is not constrained by specific network architectures.

<img src="arch.png" width="800px"/>

## Acknowledgement
Our code is built upon [KAIR](https://github.com/cszn/KAIR), [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR) and [CDC](https://github.com/xiezw5/Component-Divide-and-Conquer-for-Real-World-Image-Super-Resolution). We thank the authors for their awesome work.

## Contact
If you have any questions, please feel free to concat with me at `shallowdream555@gmail.com`.

## Citation

```BibTeX
@inproceedings{ai2024uncertainty,
  title={Uncertainty-Aware Source-Free Adaptive Image Super-Resolution with Wavelet Augmentation Transformer},
  author={Ai, Yuang and Zhou, Xiaoqiang and Huang, Huaibo and Zhang, Lei and He, Ran},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8142--8152},
  year={2024}
}
```
