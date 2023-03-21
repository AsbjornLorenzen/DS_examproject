import pandas as pd
from collections import Counter
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer 
from cleantext import clean
from cleantext.sklearn import CleanTransformer
import nltk
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
#nltk.download('punkt')
import matplotlib.pyplot as plt
import numpy as np

class preprocessor():
    def __init__(self):
        pd.options.mode.chained_assignment = None # ignore warnings
        self.ss = SnowballStemmer(language='english')
        self.lemmatizer = WordNetLemmatizer()
        
        self.cleaner = CleanTransformer(

            lower=True,           
            no_line_breaks=True,          
            no_urls=True,               
            no_emails=True,               
            no_numbers=True,               
            no_currency_symbols=False,
            replace_with_url="urltoken",
            replace_with_email="emailtoken",
            replace_with_number="numtoken",
            replace_with_currency_symbol="CURtoken",
            no_punct=True,
            lang="en")

    def clean_data(self,df,verbosity=0):
        starttime = timer()

        # Remove rows missing type
        df = df[ df['type'].notnull() ]

        #TODO: Move these to tokenize function instead 
        cleaned = self.cleaner.transform(df['content'])
        tokenized = cleaned.apply(nltk.word_tokenize)
        self.tokenized = tokenized
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

        # Use counter:
        #NOTE: Do we want to save the file as a counter object?
        df['content'] = df['content'].apply(self.to_counter)
        if (verbosity > 0):
            time4 = timer()
            print(f"Saved to df in in {round(time4-time3,3)} seconds")

        #Remove tail:
        self.remove_tail(df['content'])

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



    def getStats(self,df,when):
        #Before cleaning
        #before - pandas series with content before removing stop words and applying stemming
        words = []
        [ words.extend(el) for el in df ]
        
        c = Counter(words)
        mostcommon = c.most_common()
        word, count = zip(*mostcommon)
        word = list(word)
        count = list(count)
        
        #remove special tokens from words, and add them to the list specialTokens
        specialTokens = []
        for sT in ["numtoken","emailtoken","urltoken","numtokennumtoken","numtokennumtokennumtoken"]:
            index = word.index(sT)
            specialTokens.append((word[index],count[index]))
            del word[index]
            del count[index]
        print(specialTokens)
        
        #create first plot
        fig, ax = plt.subplots(figsize=(9,7))
        plt.plot(np.arange((10000)),count[0:10000])
        plt.grid(axis='y')
        title = f'10000 most common words({when} removing stopwords and stemming)'
        plt.title(title)
        figname1 = "10000mostcommon"+when+".png"
        plt.savefig(figname1)

        #create second plot
        #n is number of words to include in plot
        n = 50
        fig, ax = plt.subplots(figsize=(9,7))
        plt.bar(word[0:n],height=count[0:n])
        plt.xticks(rotation=90)
        #ax.set_yticks(np.arange(0, 500, 1000))
        plt.grid(axis='y')
        title = f'{n} most common words({when} removing stopwords and stemming)'
        plt.title(title)
        figname2 = str(n) + "mostcommon"+when+".png"
        plt.savefig(figname2)
        print(figname1 + " and " + figname2 + "have been updated")

        #df100A = pd.DataFrame(data={'word': word[0:100], 'count': count[0:100]})
        #with open('mostcommon100After.txt',"w") as file:
        #    file.write(df100A.to_string(header=True, index=False))
        

if __name__ == '__main__':
    p = preprocessor()
    p.bulk_preprocess(10000,'data/news_cleaned_2018_02_13.csv','data/news_cleaned_preprocessed_3')
    p.getStats(p.tokenized,"before")
    p.getStats(p.df['content'],"after")
