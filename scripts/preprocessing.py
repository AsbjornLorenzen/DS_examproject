import pandas as pd
from collections import Counter
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer 
from cleantext import clean
from cleantext.sklearn import CleanTransformer
import nltk
nltk.download('punkt')

class preprocessor():
    def __init__(self):
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

    def clean_data(self,df):
        
        #TODO: Move these to tokenize function instead 
        cleaned = self.cleaner.transform(df['content'])
        tokenized = cleaned.apply(nltk.word_tokenize)
        df['content'] = tokenized
        print(f"Vocabulary after tokenization: ",self.get_vocab_size(df))

        # Remove stopwords:
        df['content'] = df['content'].apply(self.remove_stopwords)
        print(f"Vocabulary after removal of stopwords: ",self.get_vocab_size(df))

        # Stem:
        df['content'] = df['content'].apply(self.stem)
        print(f"Vocabulary after stemming: ",self.get_vocab_size(df))

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
        self.df = pd.read_csv(filename)

    def remove_stopwords(self,tokens):
        stopwords = open('docs/stopwords.txt').read().split('\n')
        c = Counter(stopwords)
        output = [t for t in tokens if t not in c]
        return output
    
    def stem(self,tokens):
        stemmed_words = [self.ss.stem(word) for word in tokens]
        return stemmed_words

if __name__ == '__main__':
    p = preprocessor()
    p.read_data('data/newssample.txt')
    p.clean_data(p.df)
