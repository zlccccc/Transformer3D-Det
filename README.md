## Transformer3D-Det: Improving 3D Object Detection by Vote Refinement

T-CSVT 2021；Submitted in 2020.12, Accepted in 2021.7

Paper: [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/9504551/)

Video: [Bilibili](https://www.bilibili.com/video/BV19q4y127zx?spm_id_from=333.337.search-card.all.click)

------

## Introduction

Voting-based methods (e.g., VoteNet) have achieved promising results for 3D object detection.  However, the simple voting operation in VoteNet may lead to less accurate voting results that are far away from the true object centers. In this work, we propose a simple but effective 3D object detection method called Transformer3D-Det (T3D), in which we additionally introduce a transformer based vote refinement module to refine the voting results of VoteNet and can thus significantly improve the 3D object detection Specifically, our T3D framework consists of three modules: a vote generation module, a vote refinement module, and a bounding box generation module. performance. Given an input point cloud, we first utilize the vote generation module to generate multiple coarse vote clusters. Then, the clustered coarse votes will be refined by using our transformer based vote refinement module to produce more accurate and meaningful votes. Finally, the bounding box generation module takes the refined vote clusters as the input and generates the final detection results for the input point cloud. To suppress the effect of the inaccurate votes, we also propose a new non-vote loss function to train our T3D. As a result, our T3D framework can achieve better 3D object detection performance. Comprehensive experiments on two benchmark datasets ScanNetV2 and SUN RGB-D demonstrate the effectiveness of our T3D framework for 3D object detection.

![](https://github.com/zlccccc/Transformer3D-Det-zlc-3d-training-codes/blob/master/pictures/image-20220405161350191.png)

![](https://github.com/zlccccc/Transformer3D-Det-zlc-3d-training-codes/blob/master/pictures/image-20220405161428765.png)

![](https://github.com/zlccccc/Transformer3D-Det-zlc-3d-training-codes/blob/master/pictures/image-20220405161453208.png)

------

## Results

|  ScanNetV2 | Input Modality | mAP@0.25 |  mAP@0.5 |
|:----------:|:--------------:|:--------:|:--------:|
|     DSS    |    Geo + RGB   |   15.2   |    6.8   |
| F-PointNet |    Geo + RGB   |   19.8   |   10.8   |
|    GSPN    |    Geo + RGB   |   30.6   |   17.7   |
|   3D-SIS   |    Geo + MV    |   40.2   |   22.5   |
|   VoteNet  |    Geo only    |   58.7   |   33.5   |
|    HGNet   |    Geo only    |   61.3   |   34.4   |
|   MLCVNet  |    Geo only    |   64.5   |   41.4   |
|   3D-MPA   |    Geo only    |   64.2   |   49.2   |
|   H3DNet   |    Geo only    |   64.4   |   43.4   |
|   H3DNet*  |    Geo only    |   67.2   |   48.1   |
|  T3D(ours) |    Geo only    | **67.5** | **50.2** |

|   SunRGBD  | Input Modality | mAP@0.25 |
|:----------:|:--------------:|:--------:|
|     DSS    |    Geo + RGB   |   42.1   |
|  2D-driven |    Geo + RGB   |   45.1   |
|     COG    |    Geo + RGB   |   47.6   |
| F-PointNet |    Geo + RGB   |   54.0   |
|   VoteNet  |    Geo only    |   57.7   |
|   MLCVNet  |    Geo only    |   59.8   |
|   H3DNet*  |    Geo only    |   60.1   |
|    HGNet   |    Geo only    | **61.6** |
|  T3D(Ours) |    Geo only    |   60.1   |

| ScanNetV2 |    cab   |    bed   |   chair  |   sofa   |   tabl   |   door   |   wind   |   bkshf  |   pic   |   cntr   |   desk   |   curt   |   fridg  |   showr  |   toil   |   sink   |   bath   |   ofurn  |  mAP@0.5 |
|:---------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
|  VoteNet  |    8.1   |   76.1   |   67.2   |   68.8   |   42.4   |   15.3   |    6.4   |   28.0   |   1.3   |    9.5   |   37.5   |   11.6   |   27.8   |   10.0   |   86.5   |   16.8   |   78.9   |   11.7   |   33.5   |
|  MLCVNet  |   16.6   | **83.3** |   78.1   |   74.7   |   55.1   |   28.1   |   17.0   |   51.7   |   3.7   |   13.9   |   47.7   |   28.6   |   36.3   |   13.4   |   70.9   |   25.6   |   85.7   |   27.5   |   42.1   |
|  H3DNet*  |   20.5   |   79.7   |   80.1   |   79.6   |   56.2   |   29.0   |   21.3   |   45.5   |   4.2   | **33.5** |   50.6   | **37.3** | **41.4** |   37.0   |   89.1   |   35.1   | **90.2** |   35.4   |   48.1   |
|    Ours   | **22.1** |   78.1   | **82.8** | **81.0** | **61.3** | **41.2** | **23.3** | **53.8** | **9.6** |   23.4   | **58.9** |   27.2   |   37.1   | **41.8** | **92.3** | **41.7** |   87.0   | **40.1** | **50.2** |

|          Methods         | Inference time |
|:------------------------:|:--------------:|
|          VoteNet         |      0.13s     |
|          H3DNet          |      0.34s     |
| T3D (standard attention) |      0.25s     |
|            T3D           |      0.18s     |

## Install

Follow VoteNet to install the pointnet++ toolkit and download the dataset. [link](https://github.com/facebookresearch/votenet)

## Run

cd experiments/Transformer3DDet/ScanNet

​    *change Line30 and Line65 in config.yaml*

​    *change Line2 in train.sh*

sh train.sh


## Cite
```

@article{zhao2021transformer3d,
  title={Transformer3D-Det: Improving 3D object detection by vote refinement},
  author={Zhao, Lichen and Guo, Jinyang and Xu, Dong and Sheng, Lu},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={31},
  number={12},
  pages={4735--4746},
  year={2021},
  publisher={IEEE}
}

```


## More About this Codebase: (ZLC 3D Training Codes)

------

This codebase was written in the first year of my master's degree. This codebase supports a variety of 3D model training, such as RandLA-Net, PointNet, PointNet++, etc., including many previous attempts, such as HRNet + Pointnet, etc. However, because it supports too many things and the maintenance is very complex, it has been shelved for more than a year.

这套代码是我硕士一年级刚入学的时候写的，可以支持多种模型训练，如RandLA-Net，PointNet，可以做分类、语义分割、检测等等，包括了很多很多之前的尝试，如HRNet+PointNet等；但是由于它支持的东西太多，维护修改都非常复杂，所以后面的工作都是基于别人的codebase了。但是由于已有的代码迁移也很麻烦，之前的工作就只能先放这里了，希望大家能够谅解

------

**Thop (calculate flops)**

You Should download thop and change thop/profile.py line 192 and 206(dfs\_count)

​    print(prefix, module.\_get\_name(), ':')
    
​    print(prefix, module.\_get\_name(), clever_format([total_ops, total_params], '%.3f'), flush=True)
