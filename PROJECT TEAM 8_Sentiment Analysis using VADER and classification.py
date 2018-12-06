
# coding: utf-8

# In[115]:


#Sentiment Analysis using VADER AND Classification

# Importing Libraries -
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns
import pandas as pd
import numpy as np
import csv
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.stem import WordNetLemmatizer
import textblob
from textblob import TextBlob


# In[116]:


# Explore the dataset
# Loading the Dataset in to DataFrames
tweets = pd.read_csv("C:/Users/Juhi Gurbani/Desktop/FINAL_PROJECT - WEB ANALYTICS/DATA.csv", header=0,sep=',')
list(tweets.columns.values)

# We want to be determine the sentiment of a tweet on the tweet text itself;
# Hence the 'text' column is our focus.  


# In[118]:


tweets


# In[119]:


tweets['tweet_created'] = pd.to_datetime(tweets['tweet_created'])
tweets["tweet_created"] = tweets["tweet_created"].dt.date
tweets


# In[120]:


# Tweets Pre Processing
def tweet_to_words(raw_tweet):
    letters_only = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(r'http\S+)", " ",raw_tweet) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] 
    return( " ".join( meaningful_words ))

tweets['text']=tweets['text'].apply(lambda x: tweet_to_words(x))
tweets.head(45)


# In[121]:


# Graph to Plot Number of tweets of Each Airline
colors=sns.color_palette("husl", 10) 
pd.Series(tweets["Airline"]).value_counts().plot(kind = "bar",
                        color=colors,figsize=(8,6),fontsize=10,rot = 0, title = "Total No. of Tweets for each Airlines")
plt.xlabel('Airlines', fontsize=10)
plt.ylabel('No. of Tweets', fontsize=10)


# In[122]:


# Word Lemmatizer

import re, nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

def normalizer(tweet):
    only_letters = re.sub("[^a-zA-Z]", " ",tweet) 
    tokens = nltk.word_tokenize(only_letters)[1:]
    lower_case = [l.lower() for l in tokens]
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas
normalizer("@VirginAmerica plus you've added commercials to the experience... tacky.")


# In[123]:


pd.set_option('display.max_colwidth',-1) # Setting this so we can see the full content of cells
tweets['normalized_tweet']= tweets.text.apply(normalizer)
tweets[['text','normalized_tweet']]


# In[124]:


# Part of Speech

import re, nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def nouns(tweet):
    only_letters = re.sub("[^a-zA-Z]", " ",tweet) 
    tokens = nltk.word_tokenize(only_letters)
    lower_case = [l.lower() for l in tokens]
    tagged = nltk.pos_tag(tokens)
    nouns = [word for word,pos in tagged     if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')] 
    downcased = [x.lower() for x in nouns]
    joined = ''.join(downcased).encode('utf-8')
    return nouns

tweets['POS']=tweets.text.apply(nouns)
tweets[['text','POS']]


# In[125]:


# Ngrams - Unigram,Bigram,Trigram -
from nltk import ngrams
def ngrams(input_list):
    unigrams = [''.join(t) for t in list(input_list)]
    bigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:]))]
    trigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:], input_list[2:]))]
    return unigrams+bigrams+trigrams
tweets['grams'] = tweets.normalized_tweet.apply(ngrams)
tweets[['text','grams']]


# In[126]:


# VADER -

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

def print_sentiment_scores(sentence):
    
#Calling the polarity analyzer - 
    snt = analyser.polarity_scores(sentence)  
    print("{:-<30} {}".format(sentence, str(snt)))
print_sentiment_scores("United flight was a bad experience") 

#Compound value scale = -1 to 1 (-ve to +ve)
#empty list to hold our computed 'compound' VADER scores
compval1 = [ ]  

#counter
i=0 
while (i<len(tweets)):

    k = analyser.polarity_scores(tweets.iloc[i]['text'])
    compval1.append(k['compound'])
    
    i = i+1
    
#Converting sentiment values to numpy for easier usage-

compval1 = np.array(compval1)

len(compval1)
tweets['VADER score'] = compval1
tweets.head(20)


# In[127]:


# Classification of the tweets based on the Vader Score -
i = 0

#empty series to hold our predicted values
predicted_value = [ ] 

while(i<len(tweets)):
    if ((tweets.iloc[i]['VADER score'] >= 0.7)):
        predicted_value.append(1)
        i = i+1
    elif ((tweets.iloc[i]['VADER score'] >= 0) & (tweets.iloc[i]['VADER score'] < 0.7)):
        predicted_value.append(-1)
        i = i+1
    elif ((tweets.iloc[i]['VADER score'] < 0)):
        predicted_value.append(0)
        i = i+1
tweets['Sentiment']=predicted_value
tweets.head(20)


# In[128]:


#Percentage of Positive, Negative and Neutral tweets -
pos_tweets = [ tweet for index, tweet in enumerate(tweets['text']) if tweets['Sentiment'][index] > 0]
neu_tweets = [ tweet for index, tweet in enumerate(tweets['text']) if tweets['Sentiment'][index] == 0]
neg_tweets = [ tweet for index, tweet in enumerate(tweets['text']) if tweets['Sentiment'][index] < 0]
print("Percentage of positive tweets: {}%".format(len(pos_tweets)*100/len(tweets['text'])))
print("Percentage of neutral tweets: {}%".format(len(neu_tweets)*100/len(tweets['text'])))
print("Percentage of negative tweets: {}%".format(len(neg_tweets)*100/len(tweets['text'])))


# In[129]:


#groupby both airlines and rating and extract total emotions count -
print(tweets.groupby(['Airline','Sentiment']).count().iloc[:,0])


# In[130]:


# Vader
#create a graph by calling our clean_data function and then plots the total number of each tweet rating (positive,negative, or neutral)
ax = tweets.groupby(['Airline','Sentiment']).count().iloc[:,0].unstack(0).plot(kind = 'bar', title = 'Airline Ratings via Twitter')
ax.set_xlabel('Sentiment')
ax.set_ylabel('Sentiment Count')
plt.show()



# In[131]:


# Vader
#groupby by Date first making it the main index, then group by the airline, then finally the SA(sentiment) and see how many
#of each rating an airline got for each date
day_df = tweets.groupby(['tweet_created','Airline','Sentiment']).size()
day_df


# In[132]:


# Vader
# exporting the tweets dataframe to CSV
import os
tweets.to_csv('FINAL_DATA_new_Vader.csv', sep=',',encoding='utf-8',header=1)


# In[135]:


# Neglecting the neutral values and considering the positive and negative values labeled as 1 and 0 -
Tweets_Vader=pd.read_csv('FINAL_DATA_new_Vader.csv')


# In[136]:


# Linear SVM
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

count_vectorizer = CountVectorizer(ngram_range=(1,2))

tweets_text=Tweets_Vader['text']
vectorized_data = count_vectorizer.fit_transform(tweets_text)
indexed_data = hstack((np.array(range(0,vectorized_data.shape[0]))[:,None], vectorized_data))

targets=Tweets_Vader['Sentiment']

#train, test = train_test_split(indexed_data, test_size = 0.4, train_size = 0.6, random_state=1)
data_train, data_test, targets_train, targets_test = train_test_split(indexed_data, targets, test_size=0.4, random_state=0)
data_train_index = data_train[:,0]

data_train = data_train[:,1:]

data_test_index = data_test[:,0]
data_test = data_test[:,1:]

# Linear SVM
# used for Multiple classes
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
clf = OneVsRestClassifier(svm.SVC(gamma=0.01, C=100., probability=True, class_weight='balanced', kernel='linear'))
clf_output = clf.fit(data_train, targets_train)
pred_test=clf_output.predict(data_test)
#print(pred_test)
clf.score(data_test, targets_test)


# In[143]:


##Naive-Bayes 
from sklearn.naive_bayes import MultinomialNB
clf_b = MultinomialNB()
clf_b_output = clf_b.fit(data_train, targets_train)
pred_test_b=clf_b_output.predict(data_test)
#print(pred_test)
clf_b.score(data_test, targets_test)


# In[147]:


## SVM Receiver Operating Characteristic -
# calculate the fpr and tpr for all thresholds of the classification
probs = clf.predict_proba(data_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(targets_test,pred_test)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[156]:


#Naive-Bayes Receiver Operating Characteristic - 
# calculate the fpr and tpr for all thresholds of the classification
probs = clf_b.predict_proba(data_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(targets_test,pred_test_b)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[152]:


from sklearn.metrics import precision_score,recall_score,f1_score, classification_report, confusion_matrix, accuracy_score


# In[151]:


#SVM Confusion Matrix -
confusion_matrix(targets_test,pred_test)


# In[155]:


#Naive Bayes Confusion Matrix -
confusion_matrix(targets_test,pred_test_b)


# In[153]:


##Linear SVM Accuracy -
accuracy_score(targets_test,pred_test)


# In[154]:


#Naive Bayes Classifier Accuracy - 
accuracy_score(targets_test,pred_test_b)


# In[148]:


#SVM Classification Report - 
print(classification_report(targets_test,pred_test))


# In[149]:


#Naive Bayes Classification Report -
print(classification_report(targets_test,pred_test_b))

