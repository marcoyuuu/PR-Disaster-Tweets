import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

#Load and Prepare Data 
df = pd.read_csv('HumAID_maria_tweets.csv')  # Uncomment if needed

def map_labels(label):
    if label == 'not_humanitarian':
        return 'not_informative'
    elif label == 'sympathy_and_support':
        return 'neutral'
    else:
        return 'informative'

df['final_label'] = df['class_label'].apply(map_labels)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#\w+', '', tweet)
    tweet = re.sub(r'[^a-z\s]', '', tweet)
    tweet = re.sub(r'\s+', ' ', tweet).strip()

    #Tokeniz
    tokens = word_tokenize(tweet)

    #Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return ' '.join(tokens)

df['clean_tweet'] = df['tweet_text'].apply(clean_tweet)

#Split into Train, Dev, Test 
train_df = df[df['split'] == 'train']
dev_df = df[df['split'] == 'dev']
test_df = df[df['split'] == 'test']

X_train_text = train_df['clean_tweet']
y_train = train_df['final_label']

X_dev_text = dev_df['clean_tweet']
y_dev = dev_df['final_label']

X_test_text = test_df['clean_tweet']
y_test = test_df['final_label']

#Decision Tree + Bag of Words (unigrams only)
vectorizer_bow = CountVectorizer(ngram_range=(1,1), stop_words='english')
X_train_bow = vectorizer_bow.fit_transform(X_train_text)
X_dev_bow = vectorizer_bow.transform(X_dev_text)

model_dt_bow = DecisionTreeClassifier(random_state=42)
model_dt_bow.fit(X_train_bow, y_train)
y_pred_dt_bow = model_dt_bow.predict(X_dev_bow)

f1_dt_bow_macro = f1_score(y_dev, y_pred_dt_bow, average='macro')
f1_dt_bow_weighted = f1_score(y_dev, y_pred_dt_bow, average='weighted')
acc_dt_bow = accuracy_score(y_dev, y_pred_dt_bow)

#Random Forest + Bag of Words (unigrams only) 
model_rf_bow = RandomForestClassifier(random_state=42)
model_rf_bow.fit(X_train_bow, y_train)
y_pred_rf_bow = model_rf_bow.predict(X_dev_bow)

f1_rf_bow_macro = f1_score(y_dev, y_pred_rf_bow, average='macro')
f1_rf_bow_weighted = f1_score(y_dev, y_pred_rf_bow, average='weighted')
acc_rf_bow = accuracy_score(y_dev, y_pred_rf_bow)

#Decision Tree + Bag of Words (unigrams + bigrams) 
vectorizer_ngram = CountVectorizer(ngram_range=(1,2), stop_words='english')
X_train_ngram = vectorizer_ngram.fit_transform(X_train_text)
X_dev_ngram = vectorizer_ngram.transform(X_dev_text)

model_dt_ngram = DecisionTreeClassifier(random_state=42)
model_dt_ngram.fit(X_train_ngram, y_train)
y_pred_dt_ngram = model_dt_ngram.predict(X_dev_ngram)

f1_dt_ngram_macro = f1_score(y_dev, y_pred_dt_ngram, average='macro')
f1_dt_ngram_weighted = f1_score(y_dev, y_pred_dt_ngram, average='weighted')
acc_dt_ngram = accuracy_score(y_dev, y_pred_dt_ngram)

#Random Forest + Bag of Words (unigrams + bigrams) 
model_rf_ngram = RandomForestClassifier(random_state=42)
model_rf_ngram.fit(X_train_ngram, y_train)
y_pred_rf_ngram = model_rf_ngram.predict(X_dev_ngram)

f1_rf_ngram_macro = f1_score(y_dev, y_pred_rf_ngram, average='macro')
f1_rf_ngram_weighted = f1_score(y_dev, y_pred_rf_ngram, average='weighted')
acc_rf_ngram = accuracy_score(y_dev, y_pred_rf_ngram)

#Print the Results 
print("\nResults:")

print(f"Decision Tree (Bag of Words 1,1) - F1 Macro: {f1_dt_bow_macro:.4f} - F1 Weighted: {f1_dt_bow_weighted:.4f} - Accuracy: {acc_dt_bow:.4f}")
print(f"Random Forest (Bag of Words 1,1) - F1 Macro: {f1_rf_bow_macro:.4f} - F1 Weighted: {f1_rf_bow_weighted:.4f} - Accuracy: {acc_rf_bow:.4f}")
print(f"Decision Tree (Bag of Words 1,2) - F1 Macro: {f1_dt_ngram_macro:.4f} - F1 Weighted: {f1_dt_ngram_weighted:.4f} - Accuracy: {acc_dt_ngram:.4f}")
print(f"Random Forest (Bag of Words 1,2) - F1 Macro: {f1_rf_ngram_macro:.4f} - F1 Weighted: {f1_rf_ngram_weighted:.4f} - Accuracy: {acc_rf_ngram:.4f}")

#Plot F1 Macro Score Comparison
labels = [
    "DT (BOW 1,1)", "RF (BOW 1,1)", 
    "DT (BOW 1,2)", "RF (BOW 1,2)"
]
f1_macro_scores = [
    f1_dt_bow_macro, f1_rf_bow_macro, 
    f1_dt_ngram_macro, f1_rf_ngram_macro
]

plt.figure(figsize=(10,6))
bars = plt.bar(labels, f1_macro_scores, color=["skyblue", "lightgreen", "skyblue", "lightgreen"])
plt.ylim(0, 1)
plt.title("F1 Macro Score Comparison", fontsize=16)
plt.ylabel("F1 Macro Score", fontsize=14)
plt.xlabel("Model", fontsize=14)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.02, f'{height:.2f}', ha='center', va='bottom', fontsize=12)

plt.show()

#Plot Accuracy Comparison
accuracy_scores = [
    acc_dt_bow, acc_rf_bow, 
    acc_dt_ngram, acc_rf_ngram
]

plt.figure(figsize=(10,6))
bars = plt.bar(labels, accuracy_scores, color=["orange", "mediumseagreen", "orange", "mediumseagreen"])
plt.ylim(0, 1)
plt.title("Accuracy Comparison", fontsize=16)
plt.ylabel("Accuracy", fontsize=14)
plt.xlabel("Model", fontsize=14)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.02, f'{height:.2f}', ha='center', va='bottom', fontsize=12)

plt.show()