# NLP Project - Spring 2025
**Team Members:** Nate Cowan, Farid Ahmadov, Colin O'Connor, Nolan Lee, Conlan Mann

## Overview
This project implements a clickbait headline detection system using multiple different NLP algorithms. The models are trained on a labeled dataset of news headlines, where each entry is labeled as either **clickbait (1)** or **not clickbait (0)**.

This README provides detailed instructions on how to reproduce the results in a clean environment, including environment setup, data access, preprocessing steps, training procedures, and output expectations.

---

## Data Sets
Download either from this repository, or they can be found here:
- [Kaggle - News Headline Dataset](https://www.kaggle.com/datasets/amananandrai/clickbait-dataset)
- [Github - YouTube Title Dataset](https://github.com/kaustubh0201/Clickbait-Classification/blob/main/youtube_dataset.csv)

---

## Project Setup

### Environment Setup

Use Python 3.10??? or later. We recommend using a virtual environment:

```
what goes here???
```

---

## Contributions
- Nate Cowan
  - Implemented decision tree classifier
  - Wrote the introduction to the progress report, as well as the data section
  - Helped with the final presentation slides

- Colin OConnor
  - Implemented Support Vector Machine model
  - Produced the project abstract
  - Helped with final presentation slides
---

# Support Vector Machine

## Project Setup
This model was created in a Jupyter Lab environment.  
Download the `SVMclickbait.ipynb` file.  
Download both datasets: `clickbait_data.csv` and `youtube_dataset.csv`. Both files are included in the GitHub repository but can also be accessed from their links found above.

## Required Libraries
To run this project, you will need to install the following Python libraries:

- **pandas**: Used for reading CSV files and manipulating data in tabular form. Install with: **pip install pandas**

- **string**: A built-in Python module used here to help remove punctuation during text preprocessing. No installation required.

- **scikit-learn** (sklearn): A machine learning library used for feature extraction with TfidfVectorizer, splitting the dataset with train_test_split, training the model using LinearSVC, and evaluating with classification_report and confusion_matrix. Install with: **pip install scikit-learn**

- **matplotlib**: Used to visualize the confusion matrix. Install with: **pip install matplotlib**

- **seaborn**: A data visualization library built on top of matplotlib that makes plotting heatmaps cleaner and more attractive. Install with: **pip install seaborn**

## How to Run

1. Open the `SVMclickbait.ipynb` notebook in Jupyter Lab.
2. Make sure `clickbait_data.csv` and `youtube_dataset.csv` are in the same directory as the notebook.
3. Run each cell in order:
   - This will preprocess the data,
   - Train the SVM model on the clickbait dataset,
   - Evaluate the model on both the News Headline test set and the YouTube dataset.
4. Confusion matrices and classification reports will be printed and displayed as plots.

## Results
The Support Vector Machine (SVM) model was trained and tested on a dataset of news headlines. It achieved an accuracy of **95%**, with strong precision and recall for both clickbait and non-clickbait categories. This indicates that the model is highly effective when applied to the same type of data it was trained on.

To test the model's generalization ability, it was then evaluated on a separate dataset of YouTube video titles—without retraining. On this dataset, the model achieved a lower accuracy of **63%**. It performed better at identifying clickbait titles (76% recall) than non-clickbait titles (49% recall), suggesting that while clickbait patterns may transfer between domains, non-clickbait cues are more context-specific.

These results highlight that while the model generalizes moderately well to new platforms, retraining or fine-tuning with domain-specific data would likely improve performance.

## Decision Tree
This module focuses specifically on the decision tree component of the larger NLP project. For this file, you may have to edit the path to the data in the notebook, but if it is structured the same way it is in the GitHub repository, it should function. The notebook should do all preprocessing, so it can be run in its entirety.

### Required Libraries
- pandas==2.2.2
- scikit-learn==1.3.0
- matplotlib==3.8.4
```pip install pandas==2.2.2 scikit-learn==1.3.0 matplotlib==3.8.4```
---

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

