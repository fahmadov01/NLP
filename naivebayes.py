import kagglehub
import pandas as pd
import re
import string
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

#import our datasets
path = kagglehub.dataset_download("amananandrai/clickbait-dataset")
headlinesData = pd.read_csv(f"{path}/clickbait_data.csv")
youtubeData = pd.read_csv("youtube_dataset.csv")
print(youtubeData.columns)

#shuffle the data and then put it into our testing and training sets
headlinesData = headlinesData.sample(frac=1, random_state=42).reset_index(drop=False)
                                                    #currently set to 20% test and 80% training. (0.2 handles test sizing)
trainData, testData = train_test_split(headlinesData, test_size=0.2, random_state=42)

# ------------------------ Preprocessing -------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#referenced this blog post https://medium.com/@fraidoonomarzai99/naive-bayes-algorithm-in-depth-db67bb386b47 to brush up on preprocessing functionality
def preprocess(text):
    #remove all numbers from string
    text = re.sub(r'\d+', '', text)
    #remove all punctuation from the string, used https://www.geeksforgeeks.org/python-remove-punctuation-from-string/ for info on removing punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # lowercase all the characters
    text = text.lower()
    #tokenize 
    tokens = nltk.word_tokenize(text)

    #iterate through, filter out stop and short words, then lemmatize (https://www.geeksforgeeks.org/python-lemmatization-with-nltk/)
    filteredTokens = []
    for word in tokens:
        if word not in stop_words and len(word) > 1:
            lemma = lemmatizer.lemmatize(word)
            filteredTokens.append(lemma)
    return " ".join(filteredTokens)


#--------------------- stemming preprocessing -- (changed to lemmatization due to poorer performance)
#stop_words = set(stopwords.words('english'))
#stemmer = PorterStemmer()
#def preprocess(text):
    #text = text.lower()
    #text = re.sub(r'\d+', '', text)
    #text = text.translate(str.maketrans('', '', string.punctuation))
    #tokens = nltk.word_tokenize(text)
    #tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and len(word) > 1]
    #return " ".join(tokens)

#add preprocessing to the headlines
trainData['processed'] = trainData['headline'].apply(preprocess)
testData['processed'] = testData['headline'].apply(preprocess)

#turn text into TF-IDF vectors using TF-IDF vectorization, referenced https://www.geeksforgeeks.org/understanding-tf-idf-term-frequency-inverse-document-frequency/ & https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
vectorizer = TfidfVectorizer()
headlinesTrain = vectorizer.fit_transform(trainData['processed'])
headlinesTest = vectorizer.transform(testData['processed'])

labels_train = trainData['clickbait']
labels_test = testData['clickbait']

#train and evaluate our model
model = MultinomialNB()
model.fit(headlinesTrain, labels_train)


y_pred = model.predict(headlinesTest)
print(classification_report(labels_test, y_pred))


# ------------ News Confusion Matrix ----------
cm1_headlines = confusion_matrix(labels_test, y_pred)
sns.heatmap(cm1_headlines, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Clickbait', 'Clickbait'],
            yticklabels=['Not Clickbait', 'Clickbait'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix | News Dataset')
plt.show()


#Now we ran the same model on the YouTube titles dataset

youtubeData['processed'] = youtubeData['Video Title'].apply(preprocess)
youtube_vectors = vectorizer.transform(youtubeData['processed'])
youtube_labels = youtubeData['isClickbait']
youtube_preds = model.predict(youtube_vectors)

print("\n------------------ YouTube Dataset Evaluation ---------------")
print(classification_report(youtube_labels, youtube_preds))

# ------------ Youtube Confusion Matrix ----------
cm_youtube = confusion_matrix(youtube_labels, youtube_preds)
sns.heatmap(cm_youtube, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Not Clickbait', 'Clickbait'],
            yticklabels=['Not Clickbait', 'Clickbait'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix | YouTube Dataset')
plt.show()


#Bar graph (there isn't enough difference between the bars, so it is not very useful at displaying the  data, commented for now)
# precision = precision_score(labels_test, y_pred)
# recall = recall_score(labels_test, y_pred)
# f1 = f1_score(labels_test, y_pred)

# metrics = ['precision', 'recall', 'f1 Score']
# scores = [precision, recall, f1]

# plt.figure()
# plt.bar(metrics, scores, color='#083979')
# plt.ylim(0, 1)
# plt.title('Performance metrics | News Dataset')
# plt.ylabel('Score')
# plt.show()
