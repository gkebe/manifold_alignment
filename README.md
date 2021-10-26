# Manifold Alignment for GoLD Dataset

## Approach

"A cross-modality manifold alignment procedure that leverages triplet loss to jointly learn consistent, multi-modal embeddings of language-based concepts
of real-world items." [[1]](#1).

## Dataset

"The **G**r**o**unded **L**anguage **D**ataset, or GoLD is a collection of visual and English natural language data in five high-level groupings: food, home, medical, office, and tools. In these groups, 47 object classes contain 207 individual object instances. The dataset contains vision and depth images of each object from 450 different rotational views. From these, four representative ‘keyframe’ images were selected. These representative images were used to collect 16500 textual and 16500 spoken descriptions." [[2]](#2)

The dataset is available [here](https://github.com/iral-lab/gold/edit/main/README.md).

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
