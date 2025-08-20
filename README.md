# 🎬 IMDb Movie Review Sentiment Analysis with TF-IDF & ML Models

[![Releases](https://img.shields.io/badge/Releases-download-blue?logo=github)](https://github.com/naufalhaf/Sentiment-Analysis-on-Movie-Reviews/releases)

A reproducible pipeline to classify IMDb movie reviews as positive or negative. The project covers preprocessing, feature extraction with TF-IDF and CountVectorizer, training Logistic Regression and Naive Bayes, and visualizing frequent tokens with WordClouds.

---

Table of Contents
- Features
- Quick links
- Project structure
- Installation
- Data
- Preprocessing
- Feature extraction
- Model training
- Evaluation
- Visuals and examples
- CLI usage
- Extension ideas
- Contributing
- License

---

## Features
- Text cleaning: tokenization, stop-word filtering, optional stemming or lemmatization.
- Two vectorizers: CountVectorizer and TF-IDF.
- Two baseline models: Logistic Regression and Multinomial Naive Bayes.
- Train/test split, cross-validation, and standard metrics (accuracy, precision, recall, F1, ROC-AUC).
- Plotting utilities: confusion matrix, ROC curve, class balance chart.
- WordCloud generation for positive and negative reviews.
- Simple CLI and script examples for quick repro.

---

Quick links
- Releases (download and run the packaged script or executable): https://github.com/naufalhaf/Sentiment-Analysis-on-Movie-Reviews/releases  
  Download the release asset from the Releases page and execute the included file to run a packaged demo and sample models.

Badges
- Language: Python 3
- Topic tags: ai, machine-learning, nlp, sentiment-analysis, transformers

---

Project structure (example)
```
Sentiment-Analysis-on-Movie-Reviews/
├─ data/
│  ├─ imdb_train.csv
│  ├─ imdb_test.csv
├─ notebooks/
│  ├─ eda.ipynb
│  ├─ model_comparison.ipynb
├─ src/
│  ├─ preprocess.py
│  ├─ features.py
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ visualize.py
├─ requirements.txt
├─ README.md
└─ LICENSE
```

Files of interest
- src/preprocess.py — text cleaning pipeline (lowercase, strip HTML, remove punctuation, tokenization, stopwords).
- src/features.py — wrappers for CountVectorizer and TfidfVectorizer.
- src/train.py — training loop, cross-validation, model serialization.
- src/evaluate.py — compute metrics and produce confusion matrix and ROC-AUC.
- src/visualize.py — WordCloud generation and frequency bar plots.

---

Installation
- Create a Python 3.8+ virtualenv.
- Install requirements.

Example:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Typical requirements
- numpy
- pandas
- scikit-learn
- nltk
- matplotlib
- seaborn
- wordcloud

If you prefer a packaged demo, download the release asset from the Releases page and run the included script.

---

Data
- The core dataset uses IMDb movie reviews with binary labels (positive/negative).
- The repo contains a small sample in data/ for quick experiments.
- For full experiments, use a larger labeled IMDb set such as the Large Movie Review Dataset (IMDB) or an exported CSV with columns ["review", "sentiment"].
- Expected format: CSV with header and two columns: text, label.

Preprocessing
- Remove HTML tags and escape sequences.
- Convert to lowercase.
- Remove punctuation and digits (configurable).
- Tokenize with NLTK or a simple regex tokenizer.
- Remove stop words from NLTK stopword list.
- Optionally apply PorterStemmer or WordNetLemmatizer.
- Optionally keep n-grams (1-2 grams) for CountVectorizer or TF-IDF.

Example usage (Python)
```python
from src.preprocess import clean_text

sample = "<br />I loved this movie! It kept me on edge."
cleaned = clean_text(sample, lemmatize=True)
```

Feature extraction
- Two primary methods:
  - CountVectorizer (bag-of-words, optional n-grams)
  - TfidfVectorizer (tf-idf normalization)
- Keep a limited vocabulary size (e.g., max_features=20_000) to control memory.
- Use binary=True for CountVectorizer when you want presence/absence features.

Example:
```python
from src.features import build_vectorizer

tfidf, tfidf_vectorizer = build_vectorizer(method="tfidf", max_features=20000)
X = tfidf_vectorizer.fit_transform(train_texts)
```

Model training
- Train two baseline models:
  - Logistic Regression (liblinear or saga solver)
  - Multinomial Naive Bayes
- Use stratified train/test split.
- Run k-fold cross-validation (k=5) for model selection.
- Serialize models and vectorizers with joblib.

Training snippet:
```bash
python src/train.py --data data/imdb_train.csv \
                    --vectorizer tfidf \
                    --model logistic \
                    --out models/logistic_tfidf.joblib
```

Hyperparameters to tune
- For Logistic Regression: C (inverse regularization), penalty (l1, l2), solver.
- For Naive Bayes: alpha (smoothing).
- For Vectorizers: ngram_range, max_df, min_df, max_features.

Evaluation
- Use standard classification metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 score
  - ROC-AUC
- Plot a confusion matrix to inspect common error modes.
- Save metrics to CSV for experiment tracking.

Example metrics output
```
Model: Logistic Regression + TF-IDF
Accuracy: 0.88
Precision: 0.89
Recall: 0.87
F1: 0.88
ROC-AUC: 0.94
```

Visuals and examples
- WordCloud for positive reviews and negative reviews helps inspect the token-level signals.
- Bar charts show top tokens by frequency or TF-IDF weight for each class.
- Plot ROC curves to compare models.

Example WordCloud (placeholder)
![Wordcloud Example](https://raw.githubusercontent.com/amueller/word_cloud/master/examples/word_cloud.png)

Example confusion matrix (ASCII)
```
Predicted
         Neg   Pos
Actual Neg  480   40
       Pos   55  425
```

CLI usage
- Preprocess data
```bash
python src/preprocess.py --input data/imdb_train.csv --output data/imdb_train_clean.csv
```
- Vectorize and train
```bash
python src/train.py --input data/imdb_train_clean.csv \
                    --vectorizer tfidf \
                    --model logistic \
                    --cv 5 \
                    --save models/logistic_tfidf.joblib
```
- Evaluate
```bash
python src/evaluate.py --model models/logistic_tfidf.joblib \
                       --vectorizer models/tfidf_vectorizer.joblib \
                       --test data/imdb_test_clean.csv \
                       --out results/logistic_eval.json
```
- Visualize WordClouds
```bash
python src/visualize.py --input data/imdb_train_clean.csv --target sentiment
```

Packaging and Releases
- A release build bundles a small dataset, trained models, and a runnable script.
- Download releases here and run the released script to reproduce core results: https://github.com/naufalhaf/Sentiment-Analysis-on-Movie-Reviews/releases
- The release asset will include a README inside the archive that lists the exact entry point and commands to run.

Repro tips
- Fix random seed for NumPy and scikit-learn for reproducible splits and model results.
- Persist the vectorizer and model objects to avoid re-training during inference.
- Save training logs and metrics per run for experiment tracking.

Extending to deep learning
- The repo focuses on classic NLP approaches. For stronger baselines, try a transformer model:
  - Use Hugging Face Transformers (BERT-base-uncased).
  - Fine-tune on the labeled IMDb data with a classification head.
  - Compare runtime and metrics to Logistic Regression and Naive Bayes.
- You can add training scripts under src/transformers/ for model fine-tuning and use datasets and tokenizers from Hugging Face.

Good model evaluation checklist
- Use stratified sampling for test splits.
- Report class-wise precision and recall.
- Show confusion matrix and ROC-AUC.
- Compare vectorizers and models side-by-side.
- Run error analysis on false positives and false negatives to find patterns.

Contributing
- Add issues for bugs, feature requests, or experiments.
- Fork the repo, create a feature branch, and open a pull request.
- Keep unit tests for preprocessors and feature functions.
- Add notebooks for new experiments and benchmark results.

Acknowledgements and resources
- scikit-learn for vectorizers and models.
- NLTK for tokenization and stop words.
- wordcloud for visual token maps.
- Hugging Face Transformers for potential deep-learning extensions.

License
- The project uses an open-source license. See LICENSE in the repo.

Contact
- For questions or collaboration, open an issue on GitHub or submit a pull request.

<sub>Topics: ai, artificial-intelligence, bert, data-science, deep-learning, dl, gpt, huggingface, machine-learning, ml, model-training-and-evaluation, natural-language-processing-nlp, nlp, nlp-machine-learning, nltk, python, python3, sentiment-analysis-nltk, text-mining-in-python, transformers</sub>