# Rating Prediction from Amazon Food reviews (BERT)
## Introduction
A pytorch implementation of HuggingFace BERT that uses the Amazon Food Review data to predict the rating on a 5 point scale from the review by the user. The base repo for the article - (insert url)

## Objective
The repo demonstrates how to go about with fine tuning a pre-trained BERT model(I have used bert-base-uncased).

## Setup
1. Ensure that you have venv installed.
2. Create the env `python3 -m venv env`
3. Enter the env `source env/bin/activate`

## Installation Pre Reqs
1. `pip install torch`
2. `pip install transformers`
3. `pip install pandas`

## Prepare Dataset
1. Download the dataset(https://www.kaggle.com/snap/amazon-fine-food-reviews).
2. `mkdir AMAZON-DATASET`
3. unzip the dataset and place the Reviews.csv file inside AMAZON-DATASET

## Training
Run `python3 main.py`
If you have a cuda ready GPU, then the code will leverage it for training. In the event your GPU does not do the job, set the "forceCPU" config to True to verify that it works and raise an issue here.
