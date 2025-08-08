import pandas as pd
import nltk
import re
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# Load dataset
df = pd.read_csv(r"C:\Users\LEGION\Desktop\Interships\Elevvo Internship\Data set\IMDB Dataset.csv")

# Helper function to map NLTK POS tags to WordNet POS tags for lemmatization
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun if unknown

# Preprocessing function with detailed steps
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Sentence tokenize
    sentences = sent_tokenize(text)

    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    all_tokens = []

    for sentence in sentences:
        # Word tokenize
        words = word_tokenize(sentence)
        # Remove stopwords
        words = [w for w in words if w not in stop_words]
        # POS tagging
        pos_tags = pos_tag(words)

        # Lemmatize with POS
        lemmatized_words = []
        for word, tag in pos_tags:
            wn_tag = get_wordnet_pos(tag)
            lemmatized_word = lemmatizer.lemmatize(word, wn_tag)
            lemmatized_words.append(lemmatized_word)

        # Stem the lemmatized words
        stemmed_words = [ps.stem(w) for w in lemmatized_words]

        all_tokens.extend(stemmed_words)

    return ' '.join(all_tokens)

# Apply preprocessing
df['processed_review'] = df['review'].apply(preprocess_text)

# Encode labels
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['processed_review'], df['label'], test_size=0.2, random_state=42)

# Vectorize for Logistic Regression with TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Vectorize for Naive Bayes with CountVectorizer
count_vec = CountVectorizer(max_features=5000)
X_train_count = count_vec.fit_transform(X_train)
X_test_count = count_vec.transform(X_test)

# Train Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_tfidf, y_train)
y_pred_lr = lr.predict(X_test_tfidf)

# Train Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_count, y_train)
y_pred_nb = nb.predict(X_test_count)

# Evaluation
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

# Bonus: WordCloud visualization of positive and negative reviews

positive_text = ' '.join(df[df['label'] == 1]['processed_review'])
negative_text = ' '.join(df[df['label'] == 0]['processed_review'])

wc_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
wc_neg = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(negative_text)

plt.figure(figsize=(16,6))

plt.subplot(1,2,1)
plt.title("Most Frequent Positive Words")
plt.imshow(wc_pos, interpolation='bilinear')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Most Frequent Negative Words")
plt.imshow(wc_neg, interpolation='bilinear')
plt.axis('off')

plt.show()
