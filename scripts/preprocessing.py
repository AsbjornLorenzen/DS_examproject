import pandas as pd
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from cleantext import clean
from cleantext.sklearn import CleanTransformer
import nltk
nltk.download('punkt')
import matplotlib.pyplot as plt
import numpy as np

class preprocessor():
    def __init__(self):
        self.ps = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        self.cleaner = CleanTransformer(
            # Modified from clean-texts site:
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

    def clean_data(self,df):
        
        #TODO: Move these to tokenize function instead 
        cleaned = self.cleaner.transform(df['content'])
        self.tokenized = cleaned.apply(nltk.word_tokenize)
        df['content'] = self.tokenized
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
        stemmed_words = [self.ps.stem(word) for word in tokens]
        return stemmed_words
    
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
    p.read_data('data/newssample.txt')
    p.clean_data(p.df)
    p.getStats(p.tokenized,"before")
    p.getStats(p.df['content'],"after")

