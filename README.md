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
**Paper: ** https://arxiv.org/abs/2303.14189

This repository contains the code to train a [fastvit model] for [image tagging], such as classifying images, detecting objects, or predicting time-series data. This README will guide you through the steps to train and deploy it for inference.

---

## Installation

### Prerequisites

Make sure you have the following installed:
**Note:** The requirements file is in the repository itself. 

### Installing Dependencies

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/username/repository-name.git
cd repository-name
pip install -r requirements.txt
