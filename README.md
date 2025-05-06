# NLP Project - Spring 2025
- Nate Cowan

## Overview
This project implements a clickbait headline detection system using multiple different NLP algorithms. The models are trained on a labeled dataset of news headlines, where each entry is labeled as either **clickbait (1)** or **not clickbait (0)**.

This README provides detailed instructions on how to reproduce the results in a clean environment, including environment setup, data access, preprocessing steps, training procedures, and output expectations.

## Data Sets
Download wither from this repository, or they can be found here:
https://www.kaggle.com/datasets/amananandrai/clickbait-dataset
INSERT YOUTUBE LINK HERE

## Contributions
- Nate Cowan
  - Implemented decision tree classifier
  - Wrote the introduction to the progress report, as well as the data section
  - Helped with the final presentation slides

## Decision Tree
This module focuses specifically on the decision tree component of the larger NLP project. For this file, you may have to edit the path to the data in the notebook, but if it is structured the same way it is in the GitHub repository, it should function. The notebook should do all preprocessing, so it can be run in its entirety.

### Required Libraries
pandas==2.2.2
scikit-learn==1.3.0
matplotlib==3.8.4

## TF-IDF vs Word2Vec and other methods
- Data set size
  - The size of our data sets (~30000) lends itself much better to TF-IDF than something more complex like a transformer model
  - TF-IDF works well even with small text datasets because it doesn’t rely on learning word relationships, it just counts and statistics.
  - Word2Vec and similar embeddings require large corpora to learn meaningful representations.
- TF-IDF vectors combined with our models can yield high accuracy with minimal tuning, and run much faster than deep models.
- Clickbait is Lexically Obvious
  - Clickbait relies on specific words and phrases like: "You won't believe..." or "Shocking truth about..."
  - Clickbait detection is a binary classification task that often doesn't need deep context
- No need for pretraining
- Domain specific vocabulary
  - Word2Vec trained on general corpora (like Google News) might miss nuances in domain specific language (such as gaming terms) that might not have been present in our Youtube data set.

