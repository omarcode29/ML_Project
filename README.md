# ML_Project

## Files : 

### preprocess.py : preprocessing and saving the images
### object_detection_transfer_learning.ipynb : object detection model Faster R-CNN with transfer learning
### k-mean-clustering.py: test of the k-means clustering algorithm on the bounding box images
### SUIM Folder: containes the SUIM NET as implemented by Islam M. and Edge
C.[1]Islam M., Edge C., Xiao Y., Luo P. : Semantic Segmentation of Underwater Imagery: Dataset and Benchmark. Computer Vision and Pattern Recognition (cs.CV); arXiv:2004.01241 (2020)



The proposed approach involves a semi-supervised learning method for under-
water imagery semantic segmentation. The following steps describe the approch:

## 1. Data Preparation:

The first step in the proposed approach is to download and preprocess the
Segmentation of Underwater IMagery (SUIM) dataset. Specifically, 10% of
the labeled data will be used for supervised learning, while the remaining
data will be utilized as unlabeled data. To accomplish this, the PyTorch
DataLoader will be employed for loading and preprocessing the data.

## 2. Object Detection:
The next step is to train an object detection model using the labeled data.
To accomplish this, a pre-trained state of the art object detection model
Faster R-CNN is utilized to train the last layers to detect the objects in the
SUIM Dataset. This object detection model will predict the bounding boxes
for the objects present in the input images. the parameters of the pre-trained
nodel are all frozen, only the classifier layer at the end is trained. stochastic
gradient decent is used with a learning rate of 0.005 and a momentum of 0.9
.
## 3. K-Mean Clustering:
In this approach, k-mean clustering will be employed to identify the regions
of the pixels and assign pseudo-labels. The opencv library in Python will be
used to implement k-mean clustering. k is 2, either object or background.

## 4. Encoder-Decoder Architecture:
The encoder-decoder architecture, initially proposed by Islam M. and Edge
C.[1], will be used to perform semantic segmentation. The architecture can
be defined and trained in Tensorflow.

## 5. Model Training:
semi supervised Semantic Segmentation of Underwater Imagery 3
The model will be trained using both labeled and unlabeled data. The labeled
data will be utilized for supervised learning, while the unlabeled data will be
employed for semi-supervised learning. First, the object detection model will
be trained on the labeled data and used to predict bounding boxes for the
unlabeled data. Then, k-mean clustering will be used to assign pseudo-labels
to the pixels in the unlabeled data. Finally, the encoder-decoder model will
be trained on both the labeled and pseudo-labeled data.
## 6. Evaluation:
To evaluate the proposed approach, a test set of the SUIM dataset will be
utilized. Metrics like mean intersection over union (mIoU) can be used to
assess the performance of the model





