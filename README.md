# Manifold Alignment for GoLD Dataset
## Table Of Contents
 
- [Approach](#approach)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [How to use](#howto)
    * [1. Featurization](#featurization)
    * [2. Preprocessing](#preprocessing)
    * [3. Training](#training)
    * [4. Evaluation](#evaluation)
- [References](#references)
  
## Approach

"A cross-modality manifold alignment procedure that leverages triplet loss to jointly learn consistent, multi-modal embeddings of language-based concepts
of real-world items." [[1]](#1).

## Dataset

"The **G**r**o**unded **L**anguage **D**ataset, or GoLD is a collection of visual and English natural language data in five high-level groupings: food, home, medical, office, and tools. In these groups, 47 object classes contain 207 individual object instances. The dataset contains vision and depth images of each object from 450 different rotational views. From these, four representative ‘keyframe’ images were selected. These representative images were used to collect 16500 textual and 16500 spoken descriptions." [[2]](#2)

The dataset is available [here](https://github.com/iral-lab/gold/).

## Requirements
```
 dataset==1.5.0
 flair==0.9
 matplotlib==3.4.0
 numpy==1.19.2
 ordered_set==4.0.2
 pandas==1.2.4
 scikit_learn==1.0.1
 scipy==1.6.2
 skimage==0.0
 torch==1.7.1
 torchvision==0.8.2
 tqdm==4.59.0
 transformers==4.11.3
 umap==0.1.1
```

For speech featurization,
```
 fairseq==0.10.2
 fairseq.egg==info
 torchaudio==0.7.0a0+a853dff
 SoundFile==0.10.3.post1
```
## How to use
### 1. Featurization
```
cd gold_featurization
```
cp or ln -s [gold](https://github.com/iral-lab/gold/) into this folder
```
python vision_features.py
python text_vision_features_file.py
```
### 2. Preprocessing
### 3. Training
### 4. Evaluation
## References
<a id="1">[1]</a> 
Nguyen et al. (2021). 
Practical Cross-Modal Manifold Alignment for Robotic Grounded Language Learning 
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 1613--1622.


<a id="2">[2]</a> 
Kebe et al. (2021). 
A Spoken Language Dataset of Descriptions for Speech-Based Grounded Language Learning 
Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1).

Please cite these works if you use this code:

```
@inproceedings{nguyen2021practical,
  title={Practical Cross-Modal Manifold Alignment for Robotic Grounded Language Learning},
  author={Nguyen, Andre T and Richards, Luke E and Kebe, Gaoussou Youssouf and Raff, Edward and Darvish, Kasra and Ferraro, Frank and Matuszek, Cynthia},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1613--1622},
  year={2021}
}
```

```
@inproceedings{
kebe2021a,
title={A Spoken Language Dataset of Descriptions for Speech-Based Grounded Language Learning},
author={Gaoussou Youssouf Kebe and Padraig Higgins and Patrick Jenkins and Kasra Darvish and Rishabh Sachdeva and Ryan Barron and John Winder and Donald Engel and Edward Raff and Francis Ferraro and Cynthia Matuszek},
booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1)},
year={2021},
url={https://openreview.net/forum?id=Yx9jT3fkBaD}
}
```
