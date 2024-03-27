# [SalViT360](https://cyberiada.github.io/SalViT360)
Official PyTorch implementation of our paper "Spherical Vision Transformer for 360° Video Saliency Prediction" (BMVC 2023)

 [`arXiv`](https://arxiv.org/abs/2308.13004) [`BibTeX`](#CitingSalViT360) [`Project Page`](https://cyberiada.github.io/SalViT360/)

The growing interest in omnidirectional videos (ODVs) that capture the full field-of-view (FOV) has gained 360◦ saliency prediction importance in computer vision. However, predicting where humans look in 360◦ scenes presents unique challenges, including spherical distortion, high resolution, and limited labelled data. To address these challenges, we propose a novel vision-transformer-based model for omnidirectional videos named SalViT360 that leverages tangent image representations. We introduce a spherical geometry-aware spatio-temporal self-attention mechanism that is capable of effective omnidirectional video understanding. Furthermore, we present a consistency-based unsupervised regularization term for projection-based 360◦ dense-prediction models to reduce artefacts in the predictions that occur after inverse projection. Our approach is the first to employ tangent images for omnidirectional saliency prediction, and our experimental results on three ODV saliency datasets demonstrate its effectiveness compared to the state-of-the-art.

## Table of Contents
- [Setup](#setup)
- [Dataset Preparation](#dataset)
- [Training](#training)
- [Inference](#inference)

## Setup
```bash
conda create --name salvit python=3.10
conda activate salvit
conda install -c conda-forge ffmpeg
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Dataset
Dataset structure and pre-processing scripts will be shared soon.


## Training
See [training configs](https://github.com/MertCokelek/SalViT360/tree/main/configs) for details.

```bash
python main.py
    --config configs/vst-train.py
    --dataset configs/dataset_config.yml
    --wandb <online/offline/disabled>
    --gpus <your-gpu-ids>
```

## Inference
See [**Inference.iypnb**](https://github.com/MertCokelek/SalViT360/blob/main/Inference.ipynb)
####  Checkpoints are available in [**Google Drive**](https://drive.google.com/drive/folders/1cJ9ln4sH0IIdv2L-Xcyk6N17_4_7FFOW?usp=sharing)


## <a name="CitingSalViT360"></a>Citing SalViT360
If you use SalViT360 in your research, please use the following BibTeX entry.

```BibTeX
@article{cokelek2023spherical,
  title={Spherical Vision Transformer for 360-degree Video Saliency Prediction},
  author={Cokelek, Mert and Imamoglu, Nevrez and Ozcinar, Cagri and Erdem, Erkut and Erdem, Aykut},
  journal={arXiv preprint arXiv:2308.13004},
  year={2023}
}
```


