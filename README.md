# NLP Project - Spring 2025
**Team Member:** Nate Cowan

## Overview
This project implements a clickbait headline detection system using multiple different NLP algorithms. The models are trained on a labeled dataset of news headlines, where each entry is labeled as either **clickbait (1)** or **not clickbait (0)**.

This README provides detailed instructions on how to reproduce the results in a clean environment, including environment setup, data access, preprocessing steps, training procedures, and output expectations.

---

## Data Sets
Download wither from this repository, or they can be found here:
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

- Conlan Mann
  - Implemented Naive Bayes model in collaboration with Nolan Lee
  - Led final presentation slide design/structure
  - Assisted in research for 'Related Work' section of progress report, in addition to 'Next Steps'
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
   - Evaluate the model on both the test set and the YouTube dataset.
4. Confusion matrices and classification reports will be printed and displayed as plots.

## Results

### Model Trained on News Headlines

**Confusion Matrix (News):**
|                  | Predicted: Not Clickbait | Predicted: Clickbait |
|------------------|--------------------------|-----------------------|
| Actual: Not Clickbait | 3047                     | 136                   |
| Actual: Clickbait     | 182                      | 3035                  |



**Classification Report (News):**

| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| Not Clickbait  | 0.94      | 0.96   | 0.95     | 3183    |
| Clickbait      | 0.96      | 0.94   | 0.95     | 3217    |
| **Accuracy**   |           |        | **0.95** | 6400    |
| **Macro Avg**  | 0.95      | 0.95   | 0.95     | 6400    |
| **Weighted Avg** | 0.95    | 0.95   | 0.95     | 6400    |

---

### Model Tested on YouTube Titles

**Confusion Matrix (YouTube):**
|                  | Predicted: Not Clickbait | Predicted: Clickbait |
|------------------|--------------------------|-----------------------|
| Actual: Not Clickbait | 49                       | 51                    |
| Actual: Clickbait     | 24                       | 77                    |



**Classification Report (YouTube):**

| Class          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| Not Clickbait  | 0.67      | 0.49   | 0.57     | 100     |
| Clickbait      | 0.60      | 0.76   | 0.67     | 101     |
| **Accuracy**   |           |        | **0.63** | 201     |
| **Macro Avg**  | 0.64      | 0.63   | 0.62     | 201     |
| **Weighted Avg** | 0.64    | 0.63   | 0.62     | 201     |


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

