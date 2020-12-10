# Legacy-Photo-Editing-with-Learned-Noise-Prior

The github page for WACV 2021 paper: Legacy Photo Editing with Learned Noise Prior

arxiv version: https://arxiv.org/abs/2011.11309

## Legacy Photo Dataset (LP Dataset)

The proposed legacy photo dataset contains approximately 25000 legacy images crawed from the Internet. The images include the real noises. If you would like to use the dataset, please send the e-mail to Yuzhi Zhao (yzzhao2-c@my.cityu.edu.hk). A brief self-introduction and the usage of LP Dataset are required. The LP Dataset only intends for research usage and any commercial use of the LP Dataset is not allowed.

Some training samples are shown like:

<img src="./img/training_sample/37.jpg" width="1000"/><img src="./img/training_sample/7.jpg" width="1000"/><img src="./img/training_sample/15.jpg" width="1000"/>

Some masked samples are shown like:

<img src="./img/masked_sample/4.jpg" width="1000"/><img src="./img/masked_sample/7.jpg" width="1000"/><img src="./img/masked_sample/83.jpg" width="1000"/>

Some validation samples are shown like:

<img src="./img/val_sample/74.jpg" width="1000"/><img src="./img/val_sample/2346.jpg" width="1000"/><img src="./img/val_sample/2450.jpg" width="1000"/>

## Code Usage

### Requirement:

- python 3.6
- pytorch 1.0.0
- torchvision 0.2.1
- cuda 8.0
- opencv-python 4.2.0.34
- pytorch_wavelets 1.0.0

### Training:

- negan: you may alter the loss ratio for DWT-based loss for specific dataset; the discriminator of clean image domain can be removed in experiment
- iegan-inpainting-denoising: perform joint inpainting and denoising
- iegan-scribble-based-colorization: perform colorization

sorry, currently no pre-trained models provided

## Reference

If you think the paper is helpful for your research, please cite:
```bash
@inproceedings{zhao2021legacy,
  title={Legacy Photo Editing with Learned Noise Prior},
  author={Zhao, Yuzhi and Po, Lai-Man and Lin, Tingyu and Wang, Xuehui and Liu, Kangcheng and Zhang, Yujia and Yu, Win-Yin and Xian, Pengfei and Xiong, Jingjing},
  booktitle={IEEE Winter Conference on Applications of Computer Vision},
  year={2021}
}
```
