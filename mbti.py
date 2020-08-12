from nltk.stem.porter import *
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
import string


from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import word_tokenize
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
# plt.style.use('fivethirtyeight')
import seaborn as sns
import string 
import re
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from helperfunctions.PrettyConfusionMatrix import print_cm
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,confusion_matrix,precision_recall_curve
from sklearn.linear_model import LogisticRegression
import time
import warnings
from collections import defaultdict
from sklearn.metrics import roc_curve, auc
warnings.filterwarnings('ignore')
from random import choice
import lightgbm as lgb
import gc
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
import itertools


mbti = pd.read_csv('data/mbti_1.csv') # Original Dataframe
mbti_clean = mbti.copy() # Dataframe to be cleaned 
mbti_features = mbti.copy() # Dataframe to be populated with features
mbti_features.drop(columns=['posts'],inplace=True)


# Create Y Vector of classes 
y = mbti['type']
# Plot Class Count
plot = plt.figure(figsize=(20,10))
sns.countplot(y, color='#8689DE', order=y.value_counts().index)

user_posts = [re.sub(r'\|\|\|',' ',posts) for posts in mbti['posts']] 
linkless_posts = [re.sub(r'http[\S]*','',posts) for posts in user_posts] # Remove all links. 
mbti_clean['posts'] = linkless_posts mbti_clean
mbti_clean['posts'] = mbti_clean['posts'].str.lower() mbti_clean

posts_without_punct = [re.sub(r'[^a-z\s]','',posts) for posts in mbti_clean['posts']]
words = [posts.split() for posts in posts_without_punct]
stemmer = PorterStemmer()
for row in range(len(words)):
    words[row] = " ".join([stemmer.stem(word) for word in words[row] if word not in list(stop_words.ENGLISH_STOP_WORDS) and len(word) >= 3])
vectorizer = CountVectorizer(min_df=25)
word_count = vectorizer.fit_transform(words)
word_count_df = pd.DataFrame(data = word_count.toarray(), columns = vectorizer.get_feature_names())
word_count_df.insert(loc=0, column='true_type', value=mbti['type'])
word_count_df.head()

analyzer = SentimentIntensityAnalyzer()

total_compound_score = []
for i in range(len(mbti)):
    score = pd.Series([analyzer.polarity_scores(post)['compound'] for post in mbti['split_posts'].iloc[i]]).mean()
    total_compound_score.append(score)
    
mbti_features['compound_score'] = total_compound_score
total_pos_score = []
for i in range(len(posts_without_punct)):
    score = pd.Series([analyzer.polarity_scores(post)['pos'] for post in mbti['split_posts'].iloc[i]]).mean()
    total_pos_score.append(score)
mbti_features['pos_score'] = total_pos_score
total_neg_score = []
for i in range(len(posts_without_punct)):
    score = pd.Series([analyzer.polarity_scores(post)['neg'] for post in mbti['split_posts'].iloc[i]]).mean()
    total_neg_score.append(score)
mbti_features['neg_score'] = total_neg_score
total_neu_score = []
for i in range(len(posts_without_punct)):
    score = pd.Series([analyzer.polarity_scores(post)['neu'] for post in mbti['split_posts'].iloc[i]]).mean()
    total_neu_score.append(score)
mbti_features['neu_score'] = total_neu_score


def post_preprocess(df):
    i = 0
    post_list = []
    length = len(df)
    lemmatiser = WordNetLemmatizer()
    print('Processing... Be patient')
    
    for row in df.iterrows():
        # Progress bar
        i+=1
        if (i % 500 == 0 or i == length):
            print(f"Progress bar：{round(i/length*100)}%")
        posts = row[1].posts
        posts = re.sub(r'\|\|\|',' ',posts)
        posts = re.sub(r'http[\S]*', '', posts).lower()
        posts = re.sub("[^a-z\s]", ' ', posts)
        posts = ' '.join([lemmatiser.lemmatize(w) for w in posts.split(' ') if w not in stopwords.words('english')])
        
        for t in types:
            posts = posts.replace(t,'')
        post_list.append(posts)
        
    return np.array(post_list)

vectorizer_tfidf = TfidfVectorizer(min_df=0.05, max_df=0.85, analyzer='word', ngram_range=(1, 2))
vectorizer_tfidf.fit(processed_post['processed_posts'])
word_tfidf = vectorizer_tfidf.transform(processed_post['processed_posts'])
word_tfidf_df = pd.DataFrame(data = word_tfidf.toarray(), columns = vectorizer_tfidf.get_feature_names())


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, cmap=cmap) # interpolation changes the blurriness of the squares 
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.grid(b='False')

    fmt = 'd'
    thresh = cm.max() / 1.5 # threshold controls font color on opaque tile
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black") # if color is darker, use white 

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
   


def get_plot_data(probabilities):

    for types in probabilities.keys():
        model_data[types] = {'base_x':base_x, 'est_tpr':np.zeros(101), 'auc_roc':[], 'est_pr':np.zeros(101),
                             'auc_pr':[]} 
        total_splits = len(probabilities[types]) 
        for split in probabilities[types]:
            y_scores = split[0] # split[0] is the model probability of predicting a 1
            y_true = split[1] # split[1] is the true test values for that split
            fpr, tpr, thresholds = roc_curve(y_true,y_scores) # used for interpolation
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            model_data[types]['est_tpr'] += np.interp(base_x, fpr, tpr) # Add est_tpr 
            model_data[types]['est_pr'] += np.interp(base_x, recall[::-1], precision[::-1]) # Add est_precision 
            model_data[types]['auc_roc'].append(auc(fpr, tpr)) # Append AUC 
            model_data[types]['auc_pr'].append(auc(recall, precision)) # Append AUC 
        model_data[types]['est_tpr'] = model_data[types]['est_tpr'] / total_splits # Average TPRs 
        model_data[types]['est_pr'] = model_data[types]['est_pr'] / total_splits # Average TPRs 
        model_data[types]['auc_roc'] = np.mean(model_data[types]['auc_roc']) # Average AUC-ROC
        model_data[types]['auc_pr'] = np.mean(model_data[types]['auc_pr']) # Average PR
        
    return model_data

def threshold_search(y_true, y_proba):
    
    best_threshold = 0
    best_score = 0
    for threshold in [i * 0.01 for i in range(100)]:
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score

    return best_score

def model(model, X, target, nsplits=4):
	kf = StratifiedShuffleSplit(n_splits=nsplits, random_state=420)
    model_data = defaultdict()

    classes = {'EorI':['Extrovert','Introvert'], 'NorS':['Sensing', 'Intuition'],
             'TorF':['Thinking','Feeling'],'JorP':['Perceiving','Judging']}

    t = time.time()
    for col in target.columns:
        print(f"* {classes[col][0]} vs. {classes[col][1]}")
        y = target[col]
        all_auc = []
        all_accuracies = []
        f_score = []
        model_data[col] = []
        avg_cm = np.zeros(4).reshape(2,2).astype(int)
        for train, test in kf.split(X, y):
            X_train, X_test, y_train, y_test = X.loc[train], X.loc[test], y[train], y[test]
            model.fit(X_train, y_train)
            probabilities = model.predict_proba(X_test)
            score = probabilities[:, 1]
            preds = model.predict(X_test)
            model_data[col].append((score, y_test))
            all_auc.append(roc_auc_score(y_test,score))
            fscore = threshold_search(y_test,score)
            f_score.append(fscore)
            avg_cm += confusion_matrix(y_test, preds,[1,0])
        plt.figure()
        plot_confusion_matrix(avg_cm//nsplits, classes=classes[col])
        plt.show()
        print(f'Average AUC: {np.mean(all_auc):.3f}; Average best fscore: {np.mean(f_score):.3f}')
        print("\n")
    print(f"Time use:{time.time()-t:.3f}s")

    return get_plot_data(model_data)
XGB = XGBClassifier(eval_metric='auc')
xgb_tf_model = model(XGB, X_tf, target, nsplits=5)
















