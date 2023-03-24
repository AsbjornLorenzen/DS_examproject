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
import random
import pickle
import csv
import sys
import os
import glob
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
            replace_with_number="numtoken",
            replace_with_currency_symbol="CURtoken",
            no_punct=True,
            lang="en")

    # Our self-made data cleaning
    def clean_data(self,df,verbosity=0):
        starttime = timer()

        print(df)

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
        for chunk in pd.read_csv(input_file, chunksize=chunksize,nrows=nrows,engine='python'):
            processed_chunk = p.clean_data(chunk,verbosity=1)
            train, validation, test = self.split_data(processed_chunk)

            train.to_csv(output_file+'_train.csv', mode='a', index=False, header=False)
            validation.to_csv(output_file+'_validation.csv', mode='a', index=False, header=False)
            test.to_csv(output_file+'_test.csv', mode='a', index=False, header=False)

            loaded_chunks += chunksize
            endtime = timer()
            print(f"Done loading {loaded_chunks} rows in {round(endtime-starttime,3)} seconds")
    

    # Starts out by splitting into train, test, val, and then applies preprocessing using only sklearn.
    def bulk_preprocess_sk(self,nrows,input_file,output_name):
        '''Makes the tfid vector and matrix pickles'''
        print('Preprocessing data...')
        starttime = timer()
        output_dir = self.output_dir(output_name)
        loaded_chunks = 0
        chunksize = 5000

        for chunk in pd.read_csv(input_file, chunksize=chunksize,nrows=nrows,engine='python'):
            train, validation, test = self.split_data(chunk)

            train.to_csv(output_dir+'train.csv', index=False, header=False)
            validation.to_csv(output_dir+'validation.csv', index=False, header=False)
            test.to_csv(output_dir+'test.csv', index=False, header=False)

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
        words = train_df['content'].values
        stopwords = open('docs/stopwords.txt').read().split('\n')
        # The important part:
        self.tf = TfidfVectorizer(
            stop_words=stopwords, # our own stopwords list which was recommended from the course slides
            strip_accents='ascii',
            tokenizer=self.tokenize,
            max_df=0.95,
            min_df=0.05,
            max_features=2000) # 2000 words
        tfidf_matrix = self.tf.fit_transform(words)

        with open(output_dir + 'tfidf_matrix.pickle', 'wb') as handle:
            pickle.dump(tfidf_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        with open(output_dir + 'tfidf_vectorizer.pickle', 'wb') as handle:
            pickle.dump(self.tf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def random_bulk_preprocess(self,nrows,input_file,output_name):
        # Applies our custom clean_data to random rows of the df.
        print('Preprocessing data...')
        starttime = timer()

        loaded_chunks = 0
        n = 1000 #number of records in file
        s = nrows #desired sample size
        skip = sorted(random.sample(range(n),n-s))
        df = pd.read_csv(input_file,index_col=False,skiprows=skip,engine='python',usecols=range(1,16))

        column_names = [
            'id', 'domain', 'type', 'url', 'content',
            'scraped_at', 'inserted_at', 'updated_at', 'title', 'authors',
            'keywords', 'meta_keywords', 'meta_description', 'tags', 'summary'
        ]

        df.columns = column_names

        print(df.iloc[3])
        
        df = p.clean_data(df,verbosity=1)
        print('Done reading file...')
        print(df.columns)

        train, validation, test = self.split_data(df)


        output_file = self.output_dir(output_name)
        print(output_file)

        train.to_csv(output_file+'train.csv', mode='a', index=False, header=False)
        validation.to_csv(output_file+'validation.csv', mode='a', index=False, header=False)
        test.to_csv(output_file+'test.csv', mode='a', index=False, header=False)

        endtime = timer()
        print(f"Done loading {s} rows in {round(endtime-starttime,3)} seconds")

    # Creates a csv of n articles chosen at random from the corpus.
    def draw_n_samples(self,n):
        print(f"Drawing {n} samples...")
        starttime = timer()
        input_filename = 'data/news_cleaned_2018_02_13.csv'

        loaded_chunks = 0
        totn = 9408908 #number of records in corpus (according to the readme)
        skip = sorted(random.sample(range(totn),totn-n))
        df = pd.read_csv(input_filename,index_col=False,skiprows=skip,engine='python',usecols=range(1,16))

        time2 = timer()
        num_loaded = df.shape[0]
        print(f"Loaded {num_loaded} articles in {time2-starttime} seconds")

        column_names = [
            'id', 'domain', 'type', 'url', 'content',
            'scraped_at', 'inserted_at', 'updated_at', 'title', 'authors',
            'keywords', 'meta_keywords', 'meta_description', 'tags', 'summary'
        ]
        df.columns = column_names


        output_filename = 'corpustest.csv'
        df.to_csv(output_filename, index=False)

        # Remove rows missing type and content fields
        df = df[ df['type'].notnull() ]
        df = df[ df['type'] != 'unknown' ]
        df = df[ df['content'].notnull() ]

        new_size = df.shape[0]

        time3 = timer()
        print(f"Removed {num_loaded-new_size} articles with junk data in {time3-time2} seconds")

        output_filename = 'data/corpus_' + str(new_size) + '_random.csv'
        df.to_csv(output_filename, index=False)


"""         
        # Split into x and y sets and save
        train_y = train['type'].values
        train_x = train.drop(['type'],axis=1)

        test_y = test['type'].values
        test_x = test.drop(['type'],axis=1)

        val_y = validation['type'].values
        val_x = validation.drop(['type'],axis=1)

        print(f"Columns from train: {train_x.columns} test: {test_x.columns} val: {val_x.columns}")

        train_y.to_csv(output_file+'_train_y.csv', index=False)
        train_x.to_csv(output_file+'_train_x.csv', index=False)
        test_y.to_csv(output_file+'_test_y.csv', index=False)
        test_x.to_csv(output_file+'_test_x.csv', index=False)
        val_y.to_csv(output_file+'_val_y.csv', index=False)
        val_x.to_csv(output_file+'_val_x.csv', index=False) """


        

if __name__ == '__main__':
    p = preprocessor_to_text()
    #p.bulk_preprocess(10000,'data/news_cleaned_2018_02_13.csv','data/news_cleaned_preprocessed_text')
    #p.random_bulk_preprocess(1000,'data/news_cleaned_2018_02_13.csv','data/news_cleaned_preprocessed_text_random')
    p.bulk_preprocess_sk(10000,'data/news_cleaned_2018_02_13.csv','grapes')
    #p.draw_n_samples(100000)
