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

Use Python 3.10??? or later. We recommend using the virtual environment used in HW0 and other HWs.

---

## Contributions
- Nate Cowan: naco2515
  - Implemented decision tree classifier
  - Wrote the introduction to the progress report, as well as the data section
  - Helped with the final presentation slides

- Colin OConnor
  - Implemented Support Vector Machine model
  - Produced the project abstract
  - Helped with final presentation slides
    
- Conlan Mann
  - Implemented Naive Bayes Model in collaboration with Nolan Lee
  - Conducted research for 'Related Works' and helped with 'Next Steps' for the progress report
  - Assisted with design/organization of final presentation and slides

- Farid Ahmadov
  - Implemented LSTM model
  - Helped make the final presentation slides
  - Worked on the project report, especially on the 'Related Works' section
 
- Nolan Lee
  - Helped implement the Naive Bayes Model in collaboration with Conlan Mann
  - Conducted research for 'Related Works' for the progress report
  - Helped with the final presentation slides
---

## Support Vector Machine

### Project Setup
This model was created in a Jupyter Lab environment.  
Download the `SVMclickbait.ipynb` file.  
Download both datasets: `clickbait_data.csv` and `youtube_dataset.csv`. Both files are included in the GitHub repository but can also be accessed from their links found above.

### Required Libraries
To run this project, you will need to install the following Python libraries:

- **pandas**: Used for reading CSV files and manipulating data in tabular form. Install with: **pip install pandas**

- **string**: A built-in Python module used here to help remove punctuation during text preprocessing. No installation required.

- **scikit-learn** (sklearn): A machine learning library used for feature extraction with TfidfVectorizer, splitting the dataset with train_test_split, training the model using LinearSVC, and evaluating with classification_report and confusion_matrix. Install with: **pip install scikit-learn**

- **matplotlib**: Used to visualize the confusion matrix. Install with: **pip install matplotlib**

- **seaborn**: A data visualization library built on top of matplotlib that makes plotting heatmaps cleaner and more attractive. Install with: **pip install seaborn**

### How to Run

1. Open the `SVMclickbait.ipynb` notebook in Jupyter Lab.
2. Make sure `clickbait_data.csv` and `youtube_dataset.csv` are in the same directory as the notebook.
3. Run each cell in order:
   - This will preprocess the data,
   - Train the SVM model on the clickbait dataset,
   - Evaluate the model on both the News Headline test set and the YouTube dataset.
4. Confusion matrices and classification reports will be printed and displayed as plots.

### Results
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

### How to Run

1. Open the `ClickbaitDecisionTree.ipynb` notebook in Jupyter Lab.
2. Make sure `clickbait_data.csv` and `youtube_dataset.csv` are in the same directory as the notebook.
3. Run each cell in order:
   - This will preprocess the data,
   - Train the decision tree model on the clickbait dataset,
   - Evaluate the model on both the News Headline test set and the YouTube dataset.
4. Decision Tree visualizations at the bottom of the notebook
NOTE: Make sure you are looking at the correct results, I trained on both the news healines and the Youtube data, and tested both models on the other data.

### Results
The decision tree classifier achieved a good accuracy of around 90% accuracy when trained on news headlines and tested on news headlines. It performed significantly worse when tested on Youtube headlines.

## Naive Bayes

### Introduction  
In this portion of the project, we focused on building a simple yet effective model for detecting clickbait. Our thinking was that if we were to turn this project into a real product or business, it would be useful to include a low-resource model to compare against the heavier ones. We used a Naive Bayes classifier to predict whether a given headline or video title was considered clickbait. Since the structure and tone of headlines can vary across platforms like news sites and YouTube, we wanted to see if a basic model like this could generalize well across multiple different platforms.

### Why Naive Bayes?  
We decided to use Naive Bayes because it is a classic and reliable algorithm for text classification due to it's fast performance, ease of implementation, and effectiveness when paired with vectorization like TF-IDF. We chose it as a good baseline model to compare against other classifiers in our project. One main reason Naive Bayes fit well for this task is because we’re not trying to extract deep meaning or context from the text, instead we are just trying to detect patterns in word usage. Since the model assumes that each word is independent, and clickbait often relies on certain key words or phrases, this assumption actually works quite well for our usecase.

### Preprocessing  
During preprocessing, we cleaned each title by removing any numbers and punctuation, converted the whole string to lowercase, and then tokenizing the text. We also used NLTK's stopword list to filter out common words and then lemmatized the remaining words using WordNetLemmatizer. We originally tried stemming but switched to lemmatization after seeing better performance. During our presentation, someone asked whether we had taken capitalization into account, which we hadn’t initially considered. If we had more time, we would have been interested in testing whether capitalization patterns had any impact on the model's performance.

### Training and Evaluation  
As mentioned previously, we used TF-IDF to vectorize the cleaned text and trained the multimonial Naive Bayes model of the news headlines. We evaluated performance using precision, recall, F1 score, and displayed the results via confusion matrices.

### Results and Observations  
The Naive Bayes model performed strongly on the headline test set, achieving **96% accuracy**, with both classes (clickbait and non-clickbait) showing high precision and recall (see below):

```
Headline Dataset:
              precision    recall  f1-score   support

           0       0.97      0.95      0.96      3183
           1       0.95      0.97      0.96      3217

    accuracy                           0.96      6400
   macro avg       0.96      0.96      0.96      6400
weighted avg       0.96      0.96      0.96      6400
```

When applied to the YouTube dataset, performance dropped, particularly for detecting non-clickbait content; however, despite never seeing the dataset before it maintained decent precision and recall for clickbait (see below):

```
YouTube Dataset:
              precision    recall  f1-score   support

           0       0.73      0.43      0.54       100
           1       0.60      0.84      0.70       101

    accuracy                           0.64       201
   macro avg       0.66      0.64      0.62       201
weighted avg       0.66      0.64      0.62       201
```

The gap in performance reflects the challenge in trying to generalize across different platforms. While the model still picked up on strong clickbait signals that appeared in both datasets, it struggled more with recognizing what wasn't clickbait in the YouTube titles, which was likely due to differences in tone and writing style.

### Limitations & Future Work  
Since Naive Bayes relies on word frequency patterns, it can seemingly struggle when content shifting across platforms (like news vs. YouTube). Our YouTube data set was also quite small with only approximately 200 entries. If we were to continue development, as mentioned earlier, we would like to investigate including the different variables in the youtube data set. Also find a larger youtube data set, and look at capitalization. Also trying to train on youtube instead of news to see if it is more versatile

---

### Naive Bayes Setup  
- Make sure you have the following files (included in this Repo):
  - `naivebayes.py`
  - `youtube_dataset.csv'
  - `clickbait_data.csv`

### Required Libraries
- pandas==2.2.2  
- scikit-learn==1.3.0  
- matplotlib==3.8.4  
- seaborn==0.13.2  
- nltk==3.8.1  
- kagglehub==0.3.12  


### How to Run  
Once in the correct folder, us the command line to run the below command:
```bash
python3 naivebayes.py
```

---

## Long Short Term Memory (LSTM)
### Why LSTM?
We wanted a more complex model that looks at more than just the word frequency, as well as not assuming the words are all independent of each other (as is the case in clickbait). Furthermore, LSTMs are good at looking and remembering long term dependencies to find patterns which may come up in clickbait titles.
### Project Setup
To run the model, download the `lstm.ipynb file`, as well as the `clickbait_data.csv` and the `youtube_dataset.csv` files. Make sure they are in the same path as the `lstm.ipynb` file, and if not, then change the path in the code. Then, simply run the cell and wait for the program to finish running.
### Required Libraries
- **pandas**: Used to read csv file
- **numpy**: Used for numerical operations, especially to convert labels into NumPy arrays
- **re**: Used for regular expressions to clean up the file
- **sklearn**: Used to split data for training and validation, as well as to measure performance
- **tensorflow**: Used to tokenize the text
- **seaborn**: Used to plot the confusion matrix
- **matplotlib**: Used with **seaborn** to render the confusion matrix
### Implementation
We first clean up the text to remove anything that is not a lowercase letter or number. We then tokenize the words, looking at only the 10000 most common words and ignoring the rest. We use a 80-20 training and testing/validation set. We then build the model, converting the titles (of length 20) into 64 dimensional vectors. The model used was an LSTM model with 64 memory cells. The model then predicts the probability, between 0 and 1, that the title is clickbait or not. The model is training using a batchsize of 32 with 5 epochs. The model is then evaluated, and the accuracy, precision, f1-score, and recall are calculated then printed out, as well as the models confusion matrix.
### Results
The news model achieved very high precision at around 97 percent, with around 97-98 percent result in the models f1-score, precision, and recall. This shows that the news model is good at checking weather news headlines from the same dataset are clickbait or not.

We then tested the model on the Youtube data to see how it generalizes to other forms of clickbait. In this case the accuracy was significantly lower, at around 66 percent. The models precision was around 64 percent and the recall was around 73 percent, suggesting that the news model was good at correctly classifying true positives (clickbait as clickbait), but misidentified a lot of non clickbait as clickbait. It also showed that the news model is not great at being used on other forms of clickbait such as Youtube videos, which indicates that the type of clickbait on Youtube is somewhat different than the type of clickbait in news headlines.

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

