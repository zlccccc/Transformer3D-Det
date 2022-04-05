## Transformer3D-Det: Improving 3D Object Detection by Vote Refinement

------

T-CSVT 2021；Submitted in 2020.12, Accepted in 2021.7

Video: [Bilibili](https://www.bilibili.com/video/BV19q4y127zx?spm_id_from=333.337.search-card.all.click)

![](https://github.com/zlccccc/Transformer3D-Det-zlc-3d-training-codes/blob/master/pictures/image-20220405161350191.png)

![](https://github.com/zlccccc/Transformer3D-Det-zlc-3d-training-codes/blob/master/pictures/image-20220405161428765.png)

![](https://github.com/zlccccc/Transformer3D-Det-zlc-3d-training-codes/blob/master/pictures/image-20220405161453208.png)



Voting-based methods (e.g., VoteNet) have achieved promising results for 3D object detection.  However, the simple voting operation in VoteNet may lead to less accurate voting results that are far away from the true object centers. In this work, we propose a simple but effective 3D object detection method called Transformer3D-Det (T3D), in which we additionally introduce a transformer based vote refinement module to refine the voting results of VoteNet and can thus significantly improve the 3D object detection Specifically, our T3D framework consists of three modules: a vote generation module, a vote refinement module, and a bounding box generation module. performance. Given an input point cloud, we first utilize the vote generation module to generate multiple coarse vote clusters. Then, the clustered coarse votes will be refined by using our transformer based vote refinement module to produce more accurate and meaningful votes. Finally, the bounding box generation module takes the refined vote clusters as the input and generates the final detection results for the input point cloud. To suppress the effect of the inaccurate votes, we also propose a new non-vote loss function to train our T3D. As a result, our T3D framework can achieve better 3D object detection performance. Comprehensive experiments on two benchmark datasets ScanNetV2 and SUN RGB-D demonstrate the effectiveness of our T3D framework for 3D object detection.



##### Install：

Follow VoteNet to install the pointnet++ toolkit and download the dataset. [link](https://github.com/facebookresearch/votenet)

##### run：

cd experiments/Transformer3DDet/ScanNet

​    *change Line30 and Line65 in config.yaml*

​    *change Line2 in train.sh*

sh train.sh



#### ZLC 3D Training Codes

------

This code was written in my first year of master's degree. It can support a variety of 3D model training, such as RandLA-Net, PointNet, PointNet++ and so on, including many previous attempts, such as HRNet + Pointnet and so on; However, because it supports too many things and maintenance is very complex, it has been shelved for more than a year.

这套代码是我硕士一年级刚入学的时候写的，可以支持多种模型训练，如RandLA-Net，PointNet，可以做分类、语义分割、检测等等，包括了很多很多之前的尝试，如HRNet+PointNet等；但是由于它支持的东西太多，维护修改都非常复杂，所以后面的工作都是基于别人的codebase了。但是由于已有的代码迁移也很麻烦，之前的工作就只能先放这里了，希望大家能够谅解

------

**Thop (calculate flops)**

You Should download thop and change thop/profile.py line 192 and 206(dfs\_count)
    print(prefix, module.\_get\_name(), ':')
    print(prefix, module.\_get\_name(), clever_format([total_ops, total_params], '%.3f'), flush=True)
