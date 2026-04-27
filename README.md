# SENet
### 选择语言 | Language
[中文简介](#简介) | [English](#Introduction)

### 结果 | Result

<img width="2480" height="1914" alt="senet_training_curve" src="https://github.com/user-attachments/assets/ca521f60-af2d-4ea9-9c2e-b8a6ddb6bf26" />



---

## 简介
SENet 是「轻量化注意力插件」，不是「轻量化小网络」！

SENet 是由 Jie Hu、Li Shen 与 Gang Sun 于 2017 年提出的轻量化注意力增强卷积神经网络，相关成果发表于《Squeeze-and-Excitation Networks》，该模型在 ImageNet 图像分类竞赛中大幅刷新现有网络精度，以极低的计算开销显著降低模型错误率，其核心创新为**SE通道注意力模块**，能够自适应学习各特征通道的重要程度，自主强化有效特征、抑制冗余噪声特征。整体架构以主流残差网络 ResNet 为骨干，在残差卷积模块中嵌入即插即用的SE单元，依托残差结构保证深层网络梯度流通，结合全局特征建模与通道依赖建模两大核心思想，首次将**通道维度注意力机制**系统性引入CNN视觉模型。核心创新技术包含：Squeeze全局特征压缩、Excitation通道权重激励、特征逐通道缩放加权、轻量化嵌入设计不破坏原有主干网络结构。SENet 无需大幅加深网络层数即可提升表征能力，完美兼容 ResNet、VGG 等各类经典CNN架构，大幅提升模型特征筛选与语义提取能力，成为后续注意力卷积网络、轻量化模型、图像分类与检测任务的基础核心模块，广泛应用于计算机视觉各类下游任务。

## 架构
SENet的核心架构为**以残差网络为骨干、嵌入SE注意力单元的增强型深度卷积神经网络**，整体分为「卷积特征提取主干」「SE通道注意力增强模块」「残差捷径连接分支」与「全连接分类模块」四大核心部分，原论文标准输入为224×224分辨率的3通道RGB图像，最终输出对应分类类别的预测概率，具体结构与设计如下：
- **特征提取与注意力增强模块**：主干沿用ResNet经典堆叠残差块设计，通过多层3×3卷积完成浅层纹理、中层组合特征、高层语义特征的逐层提取；**SE模块固定嵌入在残差卷积主干末端、残差相加操作之前**，分为三步完成通道建模：Squeeze压缩通过全局平均池化将单通道空间特征压缩为单一全局表征；Excitation激励依靠两层全连接层与Sigmoid激活，学习通道间依赖关系，生成0~1区间的通道权重；Scale缩放通过维度复制填充与逐像素相乘，完成主干特征的通道加权优化。
- **残差连接分支**：保留原生残差捷径分支，无额外卷积与注意力操作，直接传递原始浅层特征，弥补深层卷积的信息损耗，缓解深层网络梯度消失问题，保证模型训练稳定性。
- **分类输出模块**：卷积与注意力强化后的深层特征经全局平均池化压缩维度，展平为一维特征向量，接入多层全连接映射，最终输出层维度匹配分类任务类别数（原论文ImageNet任务为1000维），输出各类别预测得分。

该架构最大优势为**SE模块轻量化、模块化、即插即用**，无需改动原有卷积与残差逻辑，仅增加少量计算量即可挖掘通道维度关键信息，自适应强化有效特征表达，成为深度学习中通道注意力机制的经典范式。
<img width="1030" height="281" alt="image" src="https://github.com/user-attachments/assets/aebfd5b3-e956-4967-8778-1dd04acc1ab7" />
<img width="1074" height="421" alt="image" src="https://github.com/user-attachments/assets/c806e643-83de-4e26-a6f7-828068a69238" />
<img width="1040" height="491" alt="image" src="https://github.com/user-attachments/assets/1f0b0122-8950-409d-8fae-ed7a1471613f" />


**注意**：我们使用的是数据集CIFAR-10，它是一个10类数据，并且不同于原文献，由于 CIFAR-10 图像尺寸（32×32）远小于原论文的 224×224，我们会对网络结构做微小适配（主要调整首层卷积核大小与步长、去除冗余下采样），但**核心SE注意力机制、残差块结构、压缩-激励-缩放完整流程**完全保留，严格复现原版SENet核心设计思想。

## 数据集
我们使用的是数据集CIFAR-10，是一个更接近普适物体的彩色图像数据集。CIFAR-10 是由Hinton 的学生Alex Krizhevsky 和Ilya Sutskever 整理的一个用于识别普适物体的小型数据集。一共包含10 个类别的RGB 彩色图片：飞机（ airplane ）、汽车（ automobile ）、鸟类（ bird ）、猫（ cat ）、鹿（ deer ）、狗（ dog ）、蛙类（ frog ）、马（ horse ）、船（ ship ）和卡车（ truck ）。每个图片的尺寸为32 × 32 ，每个类别有6000个图像，数据集中一共有50000 张训练图片和10000 张测试图片。
数据集链接为：https://www.cs.toronto.edu/~kriz/cifar.html

---

## Introduction
SENet is a lightweight attention-enhanced convolutional neural network proposed in 2017 by Jie Hu, Li Shen and Gang Sun, published in the paper *Squeeze-and-Excitation Networks*. It significantly improved the accuracy of existing CNN models on the ImageNet classification benchmark and reduced classification error with extremely low computational cost. Its core innovation is the **Squeeze-and-Excitation Channel Attention Module**, which can adaptively learn the importance of each feature channel, strengthen effective feature representation and suppress redundant noise features automatically.

Built mainly on the classic residual network ResNet, SENet plugs the lightweight SE unit into residual blocks. Combining residual connection to ensure stable gradient propagation in deep networks, it creatively modeled global feature representation and interdependence between different channels. It firstly introduced channel-wise attention into mainstream CNNs systematically. The core technologies include global average pooling for squeeze operation, fully connected bottleneck excitation, Sigmoid normalization for weight calibration, and channel-wise feature scaling. Without deepening network layers excessively, SENet greatly improves feature screening and semantic extraction ability. It is highly compatible with classic networks such as ResNet and VGG, and has become a fundamental attention module widely used in image classification, object detection and other computer vision tasks.

## Architecture
The overall architecture of SENet is an enhanced deep CNN with residual backbone and embedded SE attention units, consisting of four core parts: convolutional feature extraction backbone, SE channel attention enhancement module, residual shortcut branch and fully connected classification head. The original paper adopted 224×224 RGB images as input and output category prediction scores for 1000 classes on ImageNet. The detailed design is as follows:

- **Feature Extraction and Attention Module**: The backbone adopts stacked residual blocks to extract low-level texture, middle-level combined features and high-level semantic features step by step. The SE block is embedded after main convolution layers and before residual addition. It contains three core steps: Squeeze compresses global spatial information into a single channel descriptor via global average pooling; Excitation learns channel interdependence through two bottleneck fully connected layers and generates channel weights between 0 and 1; Scale expands channel weights to the same size as original features and completes weighted enhancement by element-wise multiplication.
- **Residual Shortcut Branch**: The shortcut path retains the original shallow feature information without additional convolution or attention calculation. It compensates for information loss in deep convolution, avoids gradient vanishing, and ensures stable model training.
- **Classification Output Module**: Attention-enhanced high-dimensional features are compressed by global average pooling and flattened into one-dimensional vectors. After fully connected layer mapping, the final output layer matches the number of classification categories and outputs prediction logits.

The most prominent advantage of SENet is the plug-and-play, lightweight design of the SE module. It does not destroy the original convolution and residual structure, and only introduces tiny computational overhead to capture channel-wise key information, establishing a classic and widely adopted paradigm for channel attention research.
<img width="1030" height="281" alt="image" src="https://github.com/user-attachments/assets/80b475b2-d202-4f1c-90e8-698178ae5ee6" />
<img width="1074" height="421" alt="image" src="https://github.com/user-attachments/assets/5a530096-3558-43f7-9fa2-fa6949cf5d95" />
<img width="1040" height="491" alt="image" src="https://github.com/user-attachments/assets/54aee17e-d075-4913-9171-70488247e60f" />


**Note:** We use the CIFAR-10 dataset with 10 classification categories. Since the 32×32 image resolution of CIFAR-10 is much smaller than the 224×224 input in the original paper, minor adjustments are made to the first convolution layer and downsampling strategy. Nevertheless, the core SE attention mechanism, residual block structure, and complete squeeze-excitation-scale pipeline are fully retained to restore the original design of SENet.

## Dataset
We used the CIFAR-10 dataset, a color image dataset that more closely approximates common objects. CIFAR-10 is a small dataset for recognizing common objects, compiled by Alex Krizhevsky and Ilya Sutskever. It contains RGB color images for 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Each image is 32 × 32 pixels, with 6000 images per category. The dataset contains 50,000 training images and 10,000 test images.

The dataset link is: https://www.cs.toronto.edu/~kriz/cifar.html

---
## 原文章 | Original article
Hu J, Shen L, Sun G. Squeeze-and-excitation networks[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 7132-7141.
