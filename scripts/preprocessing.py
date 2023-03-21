import pandas as pd
from collections import Counter
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer 
from cleantext import clean
from cleantext.sklearn import CleanTransformer
import nltk
from timeit import default_timer as timer
#nltk.download('punkt')

class preprocessor():
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

if __name__ == '__main__':
    p = preprocessor()
    df = p.read_data('data/newssample.csv')
    df = p.clean_data(df)
    p.save_df(df)
