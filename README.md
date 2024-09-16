# fastvit-training and inference

This repository is a part of the video series started by us in the channel AIML Nepal: AI Unpacked. 
We show you how to train the latest model from Apple named fastvit. This is a very small vision transformer model that is used to label images. The training process and the inference is simple and beginner friendly. 
However, we found that the steps mentioned in the official repository of Apple were a bit short and would sometimes lead to multiple issues. Please follow along the video or the instructions below to run the training and inference script. 

**Official Repository from Apple: ** https://github.com/apple/ml-fastvit


## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Training the Model](#training-the-model)
5. [Performing Inference](#performing-inference)
6. [Results](#results)
7. [Contributing](#contributing)
8. [License](#license)

---

## Introduction

The official paper for the fastvit is mentioned below if you are interested in reading the paper. 
**Paper:** https://arxiv.org/abs/2303.14189

This repository contains the code to train a [fastvit model] for [image tagging], such as classifying images, detecting objects, or predicting time-series data. 
This README will guide you through the steps to train and deploy it for inference.

---

## Installation

### Prerequisites

Make sure you have the following installed:
**Note:** The requirements file is in the repository itself. 

### Installing Dependencies

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/username/repository-name.git
cd fastvit-implementation
pip install -r requirements.txt
```
The reason we're copying all the scripts and running from this repository is that we have made some changes to the `train.py` file without which we would not have been able to run the scripts. 
*Please check the video for more details. 

### Dataset
The dataset used for training is included in the repository itself. 
Download the folder keep the structure of the folders intact as this is the recommended structure for training the model. 
The structure is the same as to the Imagenet structure that the model was trained on.
**Dataset:** `commercial_items/`

### Training the Model
Please follow the steps below to train the model:


*Since we have already installed all required libraries, we can just start with the process below. 

1. Run the code below in your terminal.
   
```bash
python -m torch.distributed.launch --nproc_per_node=1 train.py \
/path/to/ImageNet/dataset --model fastvit_t8 -b 128 --lr 1e-3 \
--native-amp --mixup 0.2 --output /path/to/save/results \
--input-size 3 256 256
```
--nproc_per_node : this is required for multiprocessing and while most of our personal devices won't have multiprocessing included; please keep this to 1.

-b : This is the batch size to be used while training. Please modify this as per your requirement. Since, the compute will be heavy for a batch size of 128, we can go as low as 8 and it wouldn't hurt the training process as much. 

Please check the accuracy and stop the training process when you feel like you've reached the accuracy needed. 

2. After the training part is over, make the required changes to the `main.py` script as advised in the docstring of the script and run the script with the command below. 

```bash
python main.py
```
