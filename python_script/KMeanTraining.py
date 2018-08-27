# -*- coding: utf-8 -*-
import os
import sys
import codecs
import json
import logging
import _pickle as cPickle
import pandas as pd
from collections import OrderedDict
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

class KMeanTextClustering:
    def __init__(self,saved_model_path,config_dict=None,train_file_path=None,tokenized_column=None):
        self.df_full_description = None
        self.record_list = []
        self.stop_word_list = []
        self.saved_model_path = saved_model_path.split(os.sep)
        self.config_dict = config_dict
        self.train_file_path = train_file_path
        self.tokenized_column = tokenized_column
        self.load_stop_word()

    def word_cut(self):
        reader = codecs.getreader('utf-8')
        with open(self.train_file_path, 'rb') as infile:
            record_list = json.load(reader(infile))
        logging.info('Dataset loaded.')
        logging.info('Word Tokenizing This May Take a While...')
        for index, record in enumerate(record_list):
            for key_column in self.tokenized_column:
                if key_column in record:
                    # pass
                    record[key_column] = word_tokenize(record[key_column], engine='mm')
            if index % 100 == 0:
                progress = '{0:.2f}'.format(float(index) / len(record_list) * 100)
                logging.info('progress : '+progress + '%')
        self.record_list = record_list
        self.remove_null()

    def remove_null(self):
        df_with_description = pd.DataFrame(self.record_list, columns=['description'])
        null_record = []
        logging.info('remove null.')
        for record_num in range(len(df_with_description)):
            if not df_with_description.iloc[record_num, 0]:
                null_record.append(record_num)
        self.df_full_description = df_with_description.drop(df_with_description.index[[null_record]])

    @staticmethod
    def split_tokenize(text):
        text = text.split(' ')
        return text

    def find_pred_dict(self,tokenized_texts):
        vectorizer = TfidfVectorizer(stop_words=self.stop_word_list,tokenizer=KMeanTextClustering.split_tokenize)
        model = KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=1)
        new_token = []
        for i in tokenized_texts:
            tmp = ' '.join(i)
            new_token.append(tmp)
        X_tfidf = vectorizer.fit_transform(new_token)
        model.fit(X_tfidf)
        term_per_clusters = OrderedDict()
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        for i in range(20):
            words_by_cluster = []
            for ind in order_centroids[i, :10]:
                words_by_cluster.append(terms[ind])
            term_per_clusters[i] = words_by_cluster
        self.save_model_and_vectorizer(model,vectorizer,term_per_clusters)

    def save_model_and_vectorizer(self,model, vectorizer,term_per_clusters):
        saved_term_per_cluster_path = self.saved_model_path+['term_per_clusters.txt']
        saved_term_per_cluster_path = os.sep.join(saved_term_per_cluster_path)
        saved_vectorizer_path = self.saved_model_path+['vectorizer.pkl']
        saved_vectorizer_path = os.sep.join(saved_vectorizer_path)
        saved_model_path = self.saved_model_path + ['kmeans.pkl']
        saved_model_path = os.sep.join(saved_model_path)
        with open(saved_model_path, 'wb')as model_file:
            cPickle.dump(model, model_file)
        model_file.close()
        with open(saved_vectorizer_path, 'wb') as fout:
            cPickle.dump(vectorizer, fout)
        fout.close()
        with open(saved_term_per_cluster_path,'w',encoding='utf-8') as fout:
            for k,v in term_per_clusters.items():
                fout.write(str(k)+','+str(v)+'\n')
        fout.close()

    def train_and_export_model(self):
        tokenized_description = self.df_full_description['description']
        logging.info('Generate Model...')
        self.find_pred_dict(tokenized_description)
        logging.info('Success.')

    def load_model(self):
        saved_model_path = self.saved_model_path+ ['kmeans.pkl']
        saved_model_path = os.sep.join(saved_model_path)
        saved_vectorizer_path = self.saved_model_path+['vectorizer.pkl']
        saved_vectorizer_path = os.sep.join(saved_vectorizer_path)
        with open(saved_model_path, 'rb') as fid:
            kmeans = cPickle.load(fid)
        fid.close()
        with open(saved_vectorizer_path, 'rb') as fid:
            vectorizer = cPickle.load(fid)
        fid.close()
        return kmeans, vectorizer

    def get_cluster(self,text):
        model,vectorizer = self.load_model()
        text = word_tokenize(text, engine='mm')
        text = ' '.join(text)
        Y = vectorizer.transform([text])
        prediction = model.predict(Y)
        return prediction

    def load_stop_word(self):
        stop_word_path = self.saved_model_path[:-1] + ['resource','stopwords-th.txt']
        stop_word_path = os.sep.join(stop_word_path)
        with open(stop_word_path,encoding='utf-8')as stop_word_file:
            for row in stop_word_file:
                row = row.rstrip()
                self.stop_word_list.append(row)
        stop_word_file.close()