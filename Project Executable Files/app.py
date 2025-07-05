import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import nltk

nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv("dataset.csv")
df.head()


"""Data Preparation"""

df.shape
df.info()

df['reviewCreatedVersion'].fillna(df['reviewCreatedVersion'].mode() [0], inplace=True)
df['replyContent'].fillna(df['replyContent'].mode() [0 ], inplace=True)
df['repliedAt'].fillna(df['repliedAt'].mode()[0], inplace=True)
df['appVersion'].fillna(df['appVersion'].mode()[0],inplace=True)

df.isnull().sum()

df.drop(["repliedAt", "replyContent", "appVersion", "reviewCreatedVersion", "reviewId", "at"], axis=1, inplace=True)

df

text_data=df["content"]
labels = df['sentiment']

train_text, test_text, y_train, y_test = train_test_split(
    text_data, labels, test_size=0.2, random_state=42)


"""Text Processing"""


def clean_text(text):
  text=text.lower()
  text=re.sub(r'\W',' ',text)
  text=re.sub(r'\s+',' ',text)
  return text

train_text.head()

train_sequence = train_text.apply(clean_text)
test_sequence = test_text.apply(clean_text)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)

"""Data Analysis"""

from wordcloud import WordCloud
import matplotlib.pyplot as plt
def plot_word_cloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(text))

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, size=20)
    plt.axis('off')
    plt.show()

plot_word_cloud(train_text, 'Original Text Word Cloud')

plot_word_cloud(train_sequence, 'Processed Text Word Cloud')

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(min_df=2,ngram_range=(1,3),max_features=10000)
vectorizer = tfidf
X_train = tfidf.fit_transform(train_sequence)
X_test = tfidf.transform(test_sequence)
vectorized_train=tfidf.fit_transform(train_sequence)
vectorized_train.shape
vectorized_test=tfidf.transform(test_sequence)
vectorized_test.shape
vectorized_train=vectorized_train.toarray()
vectorized_test-vectorized_test.toarray()
vectorized_train[0]


knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
print("KNN Classification Report:\n", classification_report(y_test, y_pred_knn))

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
print("Naive Bayes Classification Report:\n", classification_report(y_test, y_pred_nb))

rf_model = RandomForestClassifier(n_estimators=100, random_state = 42 )
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Classification Report: \n", classification_report(y_test, y_pred_rf))

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y1= model.predict(X_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

print(accuracy_score(y_train, y1))

nltk.download('punkt_tab')

# Set the nltk path manually
nltk.data.path.append('/root/nltk_data')
def predict_category(review_text):
    review_text_cleaned = clean_text(review_text)
    review_text_cleaned = remove_stopwords(review_text_cleaned)
    review_text_vectorized = vectorizer.transform([review_text_cleaned])
    prediction = model.predict(review_text_vectorized)
    return prediction[0]

sample_review = "woww "
predicted_category = predict_category(sample_review)
print(f"Predicted Category for the review '{sample_review}': {predicted_category}")

sample_review = "good"
predicted_category = predict_category(sample_review)
print(f"Predicted Category for the review '{sample_review}': {predicted_category}")




from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/input')
def input_page():
    return render_template('input.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/submit', methods=['POST'])
def output():
    content = request.form.get('content', '')
    if not content:
        return "No content provided", 400

    cleaned = clean_text(content)
    cleaned = remove_stopwords(cleaned)
    processed_content = vectorizer.transform([cleaned])
    prediction = model.predict(processed_content)[0]

    return render_template('output.html', content=content, result=prediction)

if __name__ == '__main__':
    app.run(debug=True)
