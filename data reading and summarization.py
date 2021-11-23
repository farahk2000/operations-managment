# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 18:36:45 2021

@author: farah
"""
import pandas as pd 
import numpy as np


#import carrier data 
df = pd.read_csv('C:/Users/farah/Desktop/Finaeo - BA Intern/Automation test iA Products - Sheet1.csv')
#print (df)

#remove products that are already on the platform (yes under 'on platform')
df = df[df.onplatform == 'No']
df = df.reset_index(drop=True)
#display the entire dataframe lol idk why i like to see 
pd.set_option("display.max_rows", None, "display.max_columns", None)
#print(df)

#remove insurance from name to extract term value
df['productname']= df['productname'].str.replace('Insurance ','')
# create new column with term value 
df['productterm'] = df.productname.str.split().apply(lambda x: x[1]).astype(int)

print(df['productterm'])
#rounded the term to the nearst 5 value (makes the most sense when filtering)
df['productterm']=(np.around(df.productterm.values/5, decimals=0)*5).astype(int)
print(df['productterm'])

#cleaned and organized df

#text summarizer using ntlk (product description is too long)
import bs4 as bs
import re
import nltk

scraped_data = df['productDesc'][0]
article = scraped_data

parsed_article = bs.BeautifulSoup(article,'lxml')

paragraphs = parsed_article.find_all('p')

article_text = ""

for p in paragraphs:
    article_text += p.text
# Removing Square Brackets and Extra Spaces   
article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
article_text = re.sub(r'\s+', ' ', article_text)
# Removing special characters and digits
formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )
formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

sentence_list = nltk.sent_tokenize(article_text)

stopwords = nltk.corpus.stopwords.words('english')

word_frequencies = {}
for word in nltk.word_tokenize(formatted_article_text):
    if word not in stopwords:
        if word not in word_frequencies.keys():
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1
maximum_frequncy = max(word_frequencies.values())

for word in word_frequencies.keys():
    word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
    
sentence_scores = {}
for sent in sentence_list:
    for word in nltk.word_tokenize(sent.lower()):
        if word in word_frequencies.keys():
            if len(sent.split(' ')) < 30:
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]
import heapq
summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

summary = ' '.join(summary_sentences)
print(summary)

df.replace(df['productDesc'].values,str(summary), inplace=True)
print(df['productDesc'][0])

