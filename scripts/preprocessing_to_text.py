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
#nltk.download('punkt')

class preprocessor_to_text():
    def __init__(self):
        pd.options.mode.chained_assignment = None # ignore warnings
        self.ss = SnowballStemmer(language='english')
        self.lemmatizer = WordNetLemmatizer()
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

    def clean_data(self,df,verbosity=0):
        starttime = timer()

        print(df)

        # Remove rows missing type
        df = df[ df['type'].notnull() ]

        #TODO: Move these to tokenize function instead 
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

    def tokenize(self,df):
        # TODO: Virker ikke endnu, der er noget galt med typerne. Hvis vi har tid kan vi fixe den, ellers fungerer det fint uden.
        cleaned = self.cleaner.transform(df)
        tokenized = cleaned.apply(nltk.word_tokenize)
        return tokenized

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

    def bulk_preprocess(self,nrows,input_file,output_file):
        print('Preprocessing data...')
        starttime = timer()

        loaded_chunks = 0
        chunksize = 5000

        # Applies preprocessing to one chunk at a time
        for chunk in pd.read_csv(input_file, chunksize=chunksize,nrows=nrows,engine='python'):
            processed_chunk = p.clean_data(chunk,verbosity=1)

            train, remaining = train_test_split(
                processed_chunk,test_size=0.2,random_state=42
            )

            validation, test = train_test_split(
                remaining,test_size=0.5,random_state=42
            )

            train.to_csv(output_file+'_train.csv', mode='a', index=False, header=False)
            validation.to_csv(output_file+'_validation.csv', mode='a', index=False, header=False)
            test.to_csv(output_file+'_test.csv', mode='a', index=False, header=False)

            loaded_chunks += chunksize
            endtime = timer()
            print(f"Done loading {loaded_chunks} rows in {round(endtime-starttime,3)} seconds")
    

    # Starts out by splitting into train, test, val, and then applies preprocessing using only sklearn.
    def bulk_preprocess_sk(self,nrows,input_file,output_file):
        train_output = output_file+'_train.csv'
        val_output = output_file+'_validation.csv'
        test_output = output_file+'_test.csv'

        print('Preprocessing data...')
        starttime = timer()

        loaded_chunks = 0
        chunksize = 5000

        for chunk in pd.read_csv(input_file, chunksize=chunksize,nrows=nrows,engine='python'):
            train, remaining = train_test_split(
                chunk,test_size=0.2,random_state=42
            )

            validation, test = train_test_split(
                remaining,test_size=0.5,random_state=42
            )

            train.to_csv(train_output, index=False, header=False)
            validation.to_csv(val_output, index=False, header=False)
            test.to_csv(test_output, index=False, header=False)

            loaded_chunks += chunksize
            endtime = timer()
            print(f"Done loading {loaded_chunks} rows in {round(endtime-starttime,3)} seconds")
        
        # Load train df and fit tfidf:
        # NOTE: If this file is too big, we can read it as a stream and apply tfidf to that, as in https://stackoverflow.com/questions/53754234/creating-a-tfidfvectorizer-over-a-text-column-of-huge-pandas-dataframe 
        train_df = pd.read_csv(train_output,index_col=False,engine='python',usecols=range(1,16))

        column_names = [
            'id', 'domain', 'type', 'url', 'content',
            'scraped_at', 'inserted_at', 'updated_at', 'title', 'authors',
            'keywords', 'meta_keywords', 'meta_description', 'tags', 'summary'
        ]
        train_df.columns = column_names
        words = train_df['content'].values

        # The important part:
        self.tf = TfidfVectorizer(stop_words='english',max_df=0.95,min_df=0.05,max_features=500)
        tfidf_matrix = self.tf.fit_transform(words)

        with open('data/tfidf_matrix.pickle', 'wb') as handle:
            pickle.dump(tfidf_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        with open('data/tfidf_vectorizer.pickle', 'wb') as handle:
            pickle.dump(self.tf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def random_bulk_preprocess(self,nrows,input_file,output_file):
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

        train, remaining = train_test_split(
            df,test_size=0.2,random_state=42
        )

        validation, test = train_test_split(
            remaining,test_size=0.5,random_state=42
        )

        train.to_csv(output_file+'_train.csv', mode='a', index=False, header=False)
        validation.to_csv(output_file+'_validation.csv', mode='a', index=False, header=False)
        test.to_csv(output_file+'_test.csv', mode='a', index=False, header=False)

        endtime = timer()
        print(f"Done loading {s} rows in {round(endtime-starttime,3)} seconds")

if __name__ == '__main__':
    p = preprocessor_to_text()
    #p.bulk_preprocess(50000,'data/news_cleaned_2018_02_13.csv','data/news_cleaned_preprocessed_text')
    #p.random_bulk_preprocess(1000,'data/news_cleaned_2018_02_13.csv','data/news_cleaned_preprocessed_text_random')
    #p.random_bulk_preprocess(10,'data/newssample.csv','data/news_cleaned_preprocessed_text_random')
    p.bulk_preprocess_sk(10000,'data/news_cleaned_2018_02_13.csv','data/news_cleaned_preprocessed_text_sk')
