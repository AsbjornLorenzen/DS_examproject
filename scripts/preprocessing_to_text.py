import pandas as pd
from collections import Counter
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer 
from cleantext import clean
from cleantext.sklearn import CleanTransformer
import nltk
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from random import random
import pickle
import csv
import sys
import os
import glob
import numpy as np
from gensim.models import Word2Vec
from ast import literal_eval
import string

#nltk.download('punkt')

class preprocessor_to_text():
    def __init__(self):
        pd.options.mode.chained_assignment = None # ignore warnings
        self.ss = SnowballStemmer(language='english')
        self.lemmatizer = WordNetLemmatizer()
        csv.field_size_limit(sys.maxsize) # necessary when loading large articles 
        # Define the cleaning object
        self.cleaner = CleanTransformer(
            # Modified from clean-texts site:
            lower=True,                    # lowercase text
            no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
            no_urls=True,                  # replace all URLs with a special token
            no_emails=True,                # replace all email addresses with a special token
            no_numbers=True,               # replace all numbers with a special token
            no_currency_symbols=False,
            replace_with_url="URLtoken",
            replace_with_email="EMAILtoken",
            replace_with_number="",
            replace_with_currency_symbol="CURtoken",
            no_punct=True,
            lang="en")

    # Our self-made data cleaning
    def clean_data(self,df,verbosity=0):
        starttime = timer()

        # Remove rows missing type
        df = df[ df['type'].notnull() ]

        cleaned = self.cleaner.transform(df['content'])
        tokenized = cleaned.apply(nltk.word_tokenize)
        df['content'] = tokenized
        if (verbosity > 0):
            time1 = timer()
            print(f"Vocabulary after tokenization: ",self.get_vocab_size(df))
            print(f"Tokenized in {round(time1-starttime,3)} seconds")

        # Remove stopwords:
        df['content'] = df['content'].apply(self.remove_stopwords)
        if (verbosity > 0):
            time2 = timer()
            print(f"Vocabulary after removal of stopwords: ",self.get_vocab_size(df))
            print(f"Removed stopwords in {round(time2-time1,3)} seconds")

        # Stem:
        df['content'] = df['content'].apply(self.stem)
        if (verbosity > 0):
            time3 = timer()
            print(f"Vocabulary after stemming: ",self.get_vocab_size(df))
            print(f"Stemmed in {round(time3-time2,3)} seconds")

        #Note: Here, we don't convert to counter, and we don't remove tail.

        return df

    def to_counter(self,df):
        c = Counter(df)
        return c

    def get_vocab_size(self,df):
        words = []
        [ words.extend(el) for el in df['content'] ]
        c = Counter(words)
        return len(c)

    # Custom tokenization, including stemming, to be used by sklearn
    def tokenize(self,text):
        #text = text.translate(None, string.punctuation)
        tokenized = nltk.word_tokenize(text)
        stemmed = self.stem(tokenized)
        return stemmed

    def read_data(self,filename):
        return pd.read_csv(filename)

    def remove_stopwords(self,tokens):
        stopwords = open('docs/stopwords.txt').read().split('\n')
        c = Counter(stopwords)
        output = [t for t in tokens if t not in c]
        return output
    
    def stem(self,tokens):
        stemmed_words = [self.ss.stem(word) for word in tokens]
        return stemmed_words

    def remove_tail(self, counters): #removes words that occur very infrequently
        threshold = counters.shape[0]/100 #words that on average appear in less than 1/100 of the articles
        combined_counts = sum(counters, Counter())
        words_to_delete = [k for k, v in combined_counts.items() if v <= threshold]
        for counter in counters:
            for word in words_to_delete:
                del counter[word]
        print(len(words_to_delete))
        return counters        

    def save_df(self,df):
        df.to_csv('data/newssample_preprocessed.csv')   

    def output_dir(self,name):
        output_dir = 'data/' + name + '/'
        if os.path.isdir(output_dir):
            files = glob.glob(output_dir + '*')
            for f in files:
                os.remove(f)
        else:
            os.mkdir(output_dir)
        return output_dir

    # Splits a dataframe into x and y (where y is 'type'), and splits into train, test, validation sets
    def split_data(self,df):
        column_names = df.columns

        train, remaining = train_test_split(
            df,test_size=0.2,random_state=42
        )

        validation, test = train_test_split(
            remaining,test_size=0.5,random_state=42
        )

        train.columns = validation.columns = test.columns = column_names

        return train, validation, test     

    def bulk_preprocess(self,nrows,input_file,output_file):
        print('Preprocessing data...')
        starttime = timer()

        loaded_chunks = 0
        chunksize = 5000

        # Applies preprocessing to one chunk at a time
        for chunk in pd.read_csv(input_file, chunksize=chunksize,nrows=nrows,engine='python',usecols=range(1,16)):
            column_names = [
                'id', 'domain', 'type', 'url', 'content',
                'scraped_at', 'inserted_at', 'updated_at', 'title', 'authors',
                'keywords', 'meta_keywords', 'meta_description', 'tags', 'summary'
            ]
            chunk.columns = column_names
            processed_chunk = p.clean_data(chunk,verbosity=1)
            #train, validation, test = self.split_data(processed_chunk)
            train, validation, test = self.split_data(processed_chunk)


            train.to_csv(output_file+'train.csv', mode='a', index=False, header=False)
            validation.to_csv(output_file+'validation.csv', mode='a', index=False, header=False)
            test.to_csv(output_file+'test.csv', mode='a', index=False, header=False)

            loaded_chunks += chunksize
            endtime = timer()
            print(f"Done loading {loaded_chunks} rows in {round(endtime-starttime,3)} seconds")
    

    # Starts out by splitting into train, test, val, and then applies preprocessing using only sklearn.
    def bulk_preprocess_sk(self,nrows,input_file,output_name):
        print('Preprocessing data...')
        starttime = timer()
        output_dir = self.output_dir(output_name)
        loaded_chunks = 0
        chunksize = 5000

        for chunk in pd.read_csv(input_file, chunksize=chunksize,nrows=nrows,engine='python'):
            train, validation, test = self.split_data(chunk)

            train.to_csv(output_dir+'train.csv', mode='a',index=False, header=False)
            validation.to_csv(output_dir+'validation.csv', mode='a', index=False, header=False)
            test.to_csv(output_dir+'test.csv', index=False, mode='a',header=False)

            loaded_chunks += chunksize
            endtime = timer()
            print(f"Done loading {loaded_chunks} rows in {round(endtime-starttime,3)} seconds")
        
        # Load train df and fit tfidf:
        # NOTE: If this file is too big, we can read it as a stream and apply tfidf to that, as in https://stackoverflow.com/questions/53754234/creating-a-tfidfvectorizer-over-a-text-column-of-huge-pandas-dataframe 
        train_df = pd.read_csv(output_dir+'train.csv',index_col=False,engine='python',usecols=range(1,16))
        column_names = [
            'id', 'domain', 'type', 'url', 'content',
            'scraped_at', 'inserted_at', 'updated_at', 'title', 'authors',
            'keywords', 'meta_keywords', 'meta_description', 'tags', 'summary'
        ]
        train_df.columns = column_names
        train_words = train_df['content'].values
        stopwords = open('docs/stopwords.txt').read().split('\n')
        # The important part:
        self.tf = TfidfVectorizer(
            stop_words=stopwords, # our own stopwords list which was recommended from the course slides
            strip_accents='ascii',
            tokenizer=self.tokenize,
            max_df=0.95,
            min_df=0.05,
            max_features=2000) # 2000 words
        tfidf_train_matrix = self.tf.fit_transform(train_words)

        with open(output_dir + 'tfidf_train_matrix.pickle', 'wb') as handle:
            pickle.dump(tfidf_train_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        with open(output_dir + 'tfidf_vectorizer.pickle', 'wb') as handle:
            pickle.dump(self.tf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def reservoir_sample(self,n):
        print(f"Drawing approximately {n} samples using reservoir sampling...")
        #totn = 9408908 #number of records in corpus (according to the readme)
        totn = 80000
        input_filename = 'data/corpus_100000_reservoir.csv'
        output_filename = 'data/corpus_' + str(n) + '_astrid.csv'
        try:
            os.remove(output_filename)
        except:
            pass
        starttime = timer()
        chance = n / totn
        chunksize = 10000
        loaded_chunks = 0
        loaded_lines = 0

        # Randomly selects elements from one chunk at a time
        for chunk in pd.read_csv(input_filename, chunksize=chunksize,nrows=totn,engine='python'):
            append_idx = []
            try:
                for idx in range(chunksize-1):
                    if random() < chance:
                        append_idx.append(idx)

                df = chunk.take(append_idx)
                column_names = [
                    '','id', 'domain', 'type', 'url', 'content',
                    'scraped_at', 'inserted_at', 'updated_at', 'title', 'authors',
                    'keywords', 'meta_keywords', 'meta_description', 'tags', 'summary',''
                ]
                df.columns = column_names
                # Remove rows missing type and content fields
                df = df[ df['type'].notnull() ]
                df = df[ df['type'] != 'unknown' ]
                df = df[ df['content'].notnull() ]
                df.to_csv(output_filename, mode='a', index=False, header=False)
                loaded_lines += len(append_idx)
                loaded_chunks += 1

                endtime = timer()
                print(f"Done loading {loaded_lines} rows in {round(endtime-starttime,3)} seconds")
            except Exception as e:
                print('Something went wrong while reading chunk ',loaded_chunks,e)
        time2 = timer()
        print(f"Loaded {n} articles in {time2-starttime} seconds")

    def word2vec_preprocessing_improved(self,input_dir,output_dir):
        train_df = pd.read_csv(input_dir+'train.csv',index_col=False,engine='python')#,usecols=range(1,16))
        val_df = pd.read_csv(input_dir+'validation.csv',index_col=False,engine='python')#,usecols=range(1,16))
        
        train_dim_before = train_df.shape[0]
        val_dim_before = val_df.shape[0]
        
        column_names = [
            'id', 'domain', 'type', 'url', 'content',
            'scraped_at', 'inserted_at', 'updated_at', 'title', 'authors',
            'keywords', 'meta_keywords', 'meta_description', 'tags', 'summary'
        ]
        train_df.columns = column_names
        val_df.columns = column_names
        print('Cleaning data...')
        train_df = self.clean_data(train_df)
        val_df = self.clean_data(val_df)
        
        print('Training word2vec model...')
        text = train_df['content'].values
        word2vec_model = Word2Vec(text, vector_size=300, window=5, min_count=5, workers=4, sg=0) # USE CBOG - sg=0

        text_val = val_df['content'].values

        print('Creating vectors...')
        X_train = []
        for document in text:
            document_vector = []
            for word in document:
                try:
                    word_vector = word2vec_model.wv[word]
                    document_vector.append(word_vector)
                except KeyError:
                    pass
            document_vector = np.mean(document_vector, axis=0)
            X_train.append(document_vector)

        X_val = []
        for document in text_val:
            document_vector = []
            for word in document:
                try:
                    word_vector = word2vec_model.wv[word]
                    document_vector.append(word_vector)
                except KeyError:
                    pass
            document_vector = np.mean(document_vector, axis=0)
            X_val.append(document_vector)

        print(word2vec_model.wv.most_similar("undergradu"))

        print(f"Before: {train_dim_before} After: {len(X_train)} \n Val before:{val_dim_before} after: {len(X_val)}")

        with open(output_dir + 'word2vec_train.pickle', 'wb') as handle:
            pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(output_dir + 'word2vec_val.pickle', 'wb') as handle:
            pickle.dump(X_val, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def word2vec_tfidf_preprocessing_improved(self,input_dir,output_dir):
        print('Preprocessing with word2vec and tfidf...')
        train_df = pd.read_csv(input_dir+'train.csv',index_col=False,engine='python')#usecols=range(1,16))
        val_df = pd.read_csv(input_dir+'validation.csv',index_col=False,engine='python')#,usecols=range(1,16))
        
        train_dim_before = train_df.shape
        val_dim_before = val_df.shape[0]

        print('Train dim: ',train_dim_before)
        
        column_names = [
            'id', 'domain', 'type', 'url', 'content',
            'scraped_at', 'inserted_at', 'updated_at', 'title', 'authors',
            'keywords', 'meta_keywords', 'meta_description', 'tags', 'summary'
        ]
        train_df.columns = column_names
        val_df.columns = column_names
        print(train_df.columns)
        train_df['content'] = self.cleaner.transform(train_df['content'])
        val_df['content'] = self.cleaner.transform(val_df['content'])

        text = train_df['content'].values
        text_val = val_df['content'].values

        print('Test: ',text[0:10])

        stopwords = open('docs/stopwords.txt').read().split('\n')
        tokenized_stopwords = self.tokenize(stopwords)
        print('Tokenized stopwords: ',tokenized_stopwords)
        stopwords += tokenized_stopwords

        tfidf_vectorizer = TfidfVectorizer(
            stop_words=stopwords, # our own stopwords list which was recommended from the course slides
            strip_accents='ascii',
            tokenizer=self.tokenize,
            max_df=0.95,
            min_df=0.05,
            max_features=2000) # 2000 words

        print('Fitting and transforming tfidf...')
        tfidf_vectors_train = tfidf_vectorizer.fit_transform(text)
        print('Transforming tfidf for val...')
        tfidf_vectors_val = tfidf_vectorizer.transform(text_val)
        print('tokenizing...')
        tokenizer = lambda docs: [tfidf_vectorizer.build_tokenizer()(doc) for doc in docs]
        sentences_train = tokenizer(text)
        sentences_val = tokenizer(text_val)
        print('fitting word2vec model...')
        #NOTE: We concat the train and val sentences in the word2vec, which is allowed since it is using unsupervised learning
        #word2vec_model = Word2Vec(sentences_train+sentences_val, vector_size=100, window=5, min_count=1, workers=4) 
        # Tuned below:
        word2vec_model = Word2Vec(text, vector_size=300, window=5, min_count=5, workers=4, sg=0) # USE CBOG - sg=0

        combined_vectors_train = np.zeros((len(text), tfidf_vectors_train.shape[1] + word2vec_model.vector_size))
        combined_vectors_val = np.zeros((len(text_val), tfidf_vectors_train.shape[1] + word2vec_model.vector_size))

        print('combining vectors...')
        for i in range(len(text)):
            tfidf_vector = tfidf_vectors_train[i].toarray().ravel()
            word2vec_vector = word2vec_model.wv[sentences_train[i]].mean(axis=0)
            combined_vectors_train[i] = np.concatenate((tfidf_vector, word2vec_vector))

        for i in range(len(text_val)):
            tfidf_vector = tfidf_vectors_val[i].toarray().ravel()
            word2vec_vector = word2vec_model.wv[sentences_val[i]].mean(axis=0)
            combined_vectors_val[i] = np.concatenate((tfidf_vector, word2vec_vector))

        print('combined vectors val: ',combined_vectors_val[0:3])
        print('combined vectors train: ',combined_vectors_train[0:3])

        print(f"Before: {train_dim_before} After: {len(combined_vectors_train)} \n Val before:{val_dim_before} after: {len(combined_vectors_val)}")

        with open(output_dir + 'word2vec_feat_train.pickle', 'wb') as handle:
            pickle.dump(combined_vectors_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(output_dir + 'word2vec_feat_val.pickle', 'wb') as handle:
            pickle.dump(combined_vectors_val, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Converts LIAR to a dataset with 'content' and 'type' columns, so it can be loaded by the same models 
    def load_liar(self,filename):
        type_map = {
            'pants-fire':1, 
            'false':1, 
            'barely-true':1,
            'half-true':1,
            'mostly-true':0,
            'true':0
        }
        df = pd.read_table(filename)
        output_df = pd.DataFrame(columns=['type','content'])
        types = df.iloc[:,1].map(type_map)
        text = df.iloc[:,2]
        output_df['type'] = types
        output_df['content'] = text
        output_df.to_csv('data/liar.csv')




        

if __name__ == '__main__':
    p = preprocessor_to_text()
    #p.bulk_preprocess(10000,'data/news_cleaned_2018_02_13.csv','data/news_cleaned_preprocessed_text')
    #p.random_bulk_preprocess(1000,'data/news_cleaned_2018_02_13.csv','data/news_cleaned_preprocessed_text_random')
    #p.bulk_preprocess_sk(8000,'data/corpus_10000.csv','kiwikiwi')
    #p.draw_n_samples(100000)
    #p.reservoir_sample(10000)
    #p.load_liar('data/train_liar.tsv')
    #p.word2vec_preprocessing('data/kiwi/','data/word2vectest/')
    #p.bulk_preprocess(800000,'data/corpus_1000000_reservoir.csv','data/kiwimill/')
    #p.word2vec_preprocessing_improved('data/kiwi/','data/word2vectest3/')
    #p.bulk_preprocess(80000,'data/corpus_1000000_reservoir.csv','data/kiwihun/')
    #p.bulk_preprocess_sk(80000,'data/corpus_100000_reservoir.csv','kiwikiwitest4')
    #p.word2vec_tfidf_preprocessing_improved('data/kiwihun/','data/word2vectest4/')
    #p.word2vec_preprocessing_improved('data/kiwihun/','data/word2vectest5_no_tfidf/')
    #p.bulk_preprocess(80000,'data/corpus_1000000_reservoir.csv','data/kiwimill/')
    #p.word2vec_preprocessing_improved('data/kiwimill/','data/word2vectest5/')
    #p.word2vec_preprocessing_improved('data/kiwihun/','data/word2vectest6_no_tfidf_tuned_cbow/')
    p.word2vec_preprocessing_improved('data/kiwihun/','data/word2vectest6_tuned_cbow/')



