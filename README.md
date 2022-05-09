# ObjProp

([Final paper](https://github.com/Allamrahul/ObjProp/blob/main/CS541_DL_Project.pdf))

## Introduction
This work tackles the problem for the Video Instance Segmentation. The problem is an extension of Image Instance Segmentation from the image domain to video domain. Instance segmentation is the task of detecting and masking each distinct object of interest appearing in an image. The new problem aims at simultaneous detection, segmentation and tracking of object instances in videos. This work focuses on exploring attention mechanisms to improve performance of the baseline architecture "*[Object Propagation via Inter-Frame Attentions for Temporally Stable Video Instance Segmentation](https://arxiv.org/pdf/2111.07529.pdf)*" for Video Instance Segmentation. Two new modifications are introduced, an attention neck module for improving region proposals and weighing the inter-frame affinity for mask propagation. Moreover, the techniques of sampling reference frames for mask propagation are experimented on.


## Installation
This repo is built using [mmdetection](https://github.com/open-mmlab/mmdetection). 
To install the dependencies, first clone the repository locally:
```
git clone https://github.com/anirudh-chakravarthy/objprop.git
```
Then, install PyTorch 1.1.0, torchvision 0.3.0, mmcv 0.2.12:
```
conda install pytorch==1.1.0 torchvision==0.3.0 -c pytorch
pip install mmcv==0.2.12
```
Then, install mmdetection:
```
python setup.py develop
# or "pip install -v -e ."
```
Then, install the CocoAPI for YouTube-VIS
```
conda install cython scipy
pip install git+https://github.com/youtubevos/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"
```

## Training
First, download and prepare the YouTube-VIS dataset using the [following instructions](https://github.com/youtubevos/MaskTrackRCNN#training).

To train ObjProp, run the following command:
```
python3 tools/train.py configs/masktrack_rcnn_r50_fpn_1x_youtubevos_objprop.py
```
In order to change the arguments such as dataset directory, learning rate, number of GPUs, etc, refer to the following configuration file `configs/masktrack_rcnn_r50_fpn_1x_youtubevos_objprop.py`.

## Inference
Our pretrained model can be downloaded from [Google Drive](https://drive.google.com/file/d/17kFp60O0TOFjzO8jHSXKR4YzYUYefe-C/view?usp=sharing).

To perform inference using ObjProp, run the following command:
```
python3 tools/test_video.py configs/masktrack_rcnn_r50_fpn_1x_youtubevos_objprop.py [MODEL_PATH] --out [OUTPUT_PATH.json] --eval segm
```

A JSON file with the inference results will be saved at `OUTPUT_PATH.json`. To evaluate the performance, submit the result file to the [evaluation server](https://competitions.codalab.org/competitions/20128).

## Trials

Below, we describe all our trials.

### Baseline Tuning 1
The intuition for this approach was that improving the backbone may lead to better instance segmentation of frames through better feature extraction. To this end, the feature extractor modified from ResNet-50 to ResNet-101.Furthermore, the number of fully connected layers for the tracking head and bounding box head was increased from 2 to 4. The number of convolution layers in the masking head was increased from 4 to 6. The intention was to improve the performance of these heads individually and as a form of hyperparameter tuning.

### Baseline Tuning 2
Further parameter tuning was proposed after analyzing the inference results of the baseline. In most failure cases, the object classification was incorrect, which may be a result of a weak bounding box classifier loss. The losses are weighed in the overall loss, hence the weights for the bounding box classifier loss was increased from 1 to 1.5 to put more emphasis on the classifier. Moreover, in some cases, multiple detections were identified for the a single occurrence of the object. To reduce overlapping proposals, the minimum IoU threshold was increased by a factor of 0.1.

### Attention Neck
This module is a Spatiotemporal attention layer placed between the FPN and the RPN. The authors hypothesize that introducing an attention map as an input to the RPN should improve the overall score. The shape of input has to be maintained in this module to avoid any other changes in the overall architecture. The architecture can be seen  below:

![image](https://user-images.githubusercontent.com/18104407/167492576-18bc5aa4-8759-4f97-8d74-3977d2ddbf28.png)

The attention neck takes in two inputs which are feature maps of the current and the reference frames from the FPN, respectively. A feature map has the dimensions C × W × H where C is the number of channels, W is the width of the image, and H is the height of the image. After performing convolution Conv1, the two feature maps are multiplied following a softmax to form a correlation map. The current frame is reshaped to W H × C to account for this. The resulting correlation map has the dimensions W H × W H. The attention map is calculated with the correlation map by performing another matrix multiplication with the result of a convolution on the current frame. The output from Conv2 gives C × W × H, so its multiplication with the correlation map results in C × W H. This is reshaped again to C × W × H, multiplied with a learnable parameter (γ) and added again to the current frame. Each convolution layer is followed by ReLU activation. The attention neck can seen below.

![image](https://user-images.githubusercontent.com/18104407/167492717-fd122178-7fd5-4c06-bdf3-cd5a2476644b.png)

### Weighted Inter-Frame Affinity
This is a novel approach as addition to the propagation head where multiple inter-frame affinity matrices (Figure 3) are weighed to produce a single output. 

![image](https://user-images.githubusercontent.com/18104407/167493332-f25e1829-5fb8-4dcb-9919-0d5de474f023.png)

The aim is to produce weights that can improve attention on particular classes if present in the frame to improve mask propagation. The input to this module is the features of the current and reference frame. Each has the dimension B × F × H × W where B is the batch size and F is the number of features in the frame. The reference frame is reshaped to the dimen- sion B × HW × F . The weights for inter-frame affinity are initialized (different for each batch) with the dimension B × F<sub>t</sub> × F<sub>t−δ</sub>. However, the number of features in both frames is equal (F<sub>t</sub> = F<sub>t−δ</sub>). A batch matrix multiplication is performed first between the reference frame and the weights, then the result and the current frame. The output has the dimensions B × Ft × Ft−δ . This does not change the rest of the architecture of the propagation head.

## Results

### Training Losses

The loss trend can be seen in Figure 4, where the following losses have been plotted:
- *RPN Classification Loss*: Provides information in regards to the model’s ability to confidently predict the existence of an object.
- *RPN B-Box Loss*: Provides information in regards to the model’s ability to confidently predict the dimen- sions of the region proposal.
- *Match Loss*: Provides information in regards to the model’s ability to effectively track objects across frames.
- *Classification Loss*: Provides information in regards to the model’s ability to classify an object as one of the 41 classes.
- *B-Box Loss*: Provides information in regards to the model’s ability to predict the dimensions of an object
- *Total Loss*: Sum of all the above losses

![image](https://user-images.githubusercontent.com/18104407/167494971-88d59898-8e16-4ba9-be52-4b3ffc86f899.png)

### Comparitive Analysis
In this section, the associated empirical performance are discussed based on performance metrics and inferred for a comparative study (Table 1).

![image](https://user-images.githubusercontent.com/18104407/167495682-5dae27f7-696c-4bb0-bc98-80be2b5aacdd.png)

In **Baseline Tuning 1**, the performance degrades across all validation metrics. Perhaps, the increased backbone complexity ((∼ 27 million) additional parameters) caused the model to learn information specific to the training data and not generalize well enough for the validation set and consequently, the real world.

In **Baseline Tuning 2**, the performance degrades across all validation metrics. Perhaps, increasing the parameter values of MaxIoUAssigner results in missed detections. For example, if the region proposals just cover a part of the object but not the whole object, the computed IoU with the ground truth bounding box would be less even though the object is present around that area. Additionally, increasing the loss weight of the bounding box classifier, should have ideally penalized the model more for any misclassifications, which forces the model to further tweak the weights to get the associated loss under control. However, since there is a degrade in performance, the authors believe the threshold change to MaxIoUAssigner may have countered the impact of to the loss weighing.

In **Baseline with Ordered Pairs**, the performance degrades across all validation metrics. In general, the movement of an object from the previous frame to the current frame is not significant, unless the object is moving too fast. Based on the latter statement, if the previous frame is picked as a reference frame, the model fails to learn weights capable of generalizing to cases where there is a significant change from the previous frame to the current frame. Hence, this could be the reason why the model fails to improve. Instead, if the reference frame were random (baseline), there would be a significant difference between the reference and current frames, which forces the weights to adapt to be able to handle these situations.

In **Additional Training**, metrics AP75 and mAP slightly improve over the baseline model. Perhaps, the increased training finetuned the model parameters further, which as a consequence bumped the AP75 and mAP scores.

In **Attention Neck**, the performance degrades across all validation metrics. Both convolutions feature a single layer with 256 3 × 3 filters with a ReLU activation. Perhaps, the Conv1 and Conv2 layers are not deep enough to learn anything meaningful from the current frame and reference frame feature maps, which as a consequence is affecting the attention map generated by the Neck Attention module.

In **Attention Neck with Ordered Pairs**, the performance although degrades in comparison to baseline, it at least performs better than the Attention Neck trial. Perhaps, The attention neck estimates the global correlation map between the successive frames and transfers it to the attention map. Added with the attention information, the new features may enhance the response of the instance for predefined categories.

In **Weighted Inter-frame Affinity**, the proposed approach had to be tweaked during execution. The weight size was fixed in implementation for training. However, this was problematic in testing since the batch size is 1. Thus, the layers in the weighted inter-frame affinity approach were pruned during testing. This may have compromised the result. This, as a consequence, degraded the model by far the largest.

### Visual Analysis

In this section, scenarios where the model performs well are discussed. At the same time, the scenarios where the model fails to either segment, track or detect objects in a frame sequence are also discussed. In Figure 5, in the first sequence of images, there are two human instances present and the instance color consistency
confirms is maintained throughout the sequence of frames. In the second sequence of images, the human instances and the motorcycle instances are tracked accurately. The instance color consistency confirms this. The same applies to the third sequence of images as well. However, there are a multitude of scenarios where the model does not perform well. 

![image](https://user-images.githubusercontent.com/18104407/167498407-5d569c5d-89ed-42c4-aa84-7b8a72c04ab1.png)

For instance, in Figure 6, in the first sequence of images, the model misclassifies a bear as an ape but also creates a new ape instance after the ape gets occluded by the tree in frame t. In the second sequence of images, even though the model detects two human instances at frame t − ∆, the associated masks are slightly misplaced. At frame t, the model incorrectly loses track of the previous instance and creates a new instance even though the instance itself is unchanged. That being said, it at least gets the mask for each instance right. At frame t + ∆, the model predicts a new instance, which is not the case. There are two human instances but the model incorrectly detects three. Finally, in the third image sequence, the model accurately detects the object instances and their associated masks. But in frame t, the model incorrectly detects a new truck instance but recovers back at frame t + ∆, by maintaining the instance color consistency for the truck
instance between frame t − ∆ and t + ∆.

![image](https://user-images.githubusercontent.com/18104407/167498467-f2ffb8c5-034b-45c6-bbf1-4dc24d07ad71.png)

Additionally in Figure 7, in the first sequence of frames, the model incorrectly detects an inanimate non-object as an object in frame t and misclassifies it as a frog. In the second sequence of frames, the model incorrectly identifies a dog as a giant panda and a mouse at frames t − ∆ and t. It corrects for this at frame t + ∆. Finally, in the third sequence of images, the model correctly predicts the instances (lizard and hand) at frame t−∆. However, at frame t, the model fails completely. The model detects two more instances (cat and mouse), which are not present in the image. On top of this, the instance masks for both cat and mouse are sparse in nature. In essence, the model thinks there is a cat and a mouse instance and the hand is occluding them, which resulted in the sparse instance masks, which is the wrong conclusion. Finally, at frame t + ∆, the model still identifies the hand as the same instance but incorrectly identifies the lizard as a new instance.

![image](https://user-images.githubusercontent.com/18104407/167498654-3846dee2-2390-43a6-a535-b1e677bc34de.png)

## Conclusions and Future Work

In this work, although the authors find that adding Attention Neck with Ordered Pairs module in the baseline network degrades its performance compared to baseline, it at least performs better than the Attention Neck trial. It is speculated that the attention neck estimates the global correlation map between the successive frames and transfers it to the attention map. This could be a research direction in future. Weighted Inter-Frame Affinity also decreases the overall performance of the model. Adding weights may reduces the learning capacity of the base network and on the run-time does not actually lead to any better performance. However, this implementation can be changed to having separate weight for each class rather than basing its dimension on batch. Random sampling of frames may perform better due to the fact that random sampling generalizes very well since some objects maybe occluded in preceding and subsequent frames. Another interesting observation noted is that additional training improves AP75 values leading to the increase in overall mAP. This paper advances the field by introducing and implementing two different modules and its effect on the ObjProp architecture. Though, the goal of this work was to implement it on state-of-the-art in the domain which did not achieve a better performance. However, it should also be acknowledged the research could not be refined due to time constraints of the course, CS541. Yet, the results of the present work may prove useful to guide future work in this field of research.


## License
ObjProp is released under the [Apache 2.0 license](LICENSE).

## Citation
```
@article{Chakravarthy2021ObjProp,
  author = {Anirudh S Chakravarthy and Won-Dong Jang and Zudi Lin and Donglai Wei and Song Bai and Hanspeter Pfister},  
  title = {Object Propagation via Inter-Frame Attentions for Temporally Stable Video Instance Segmentation},
  journal = {CoRR},
  volume = {abs/2111.07529},
  year = {2021},
  url = {https://arxiv.org/abs/2111.07529}
}
```
