import preprocessing as prep
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
class data_explorer():
    def getMostCommon(self,df,when):
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
        

        with open("expl_output/exploration_stats.txt",'w') as f:
            print("Special tokens:",file=f)
            print(specialTokens,file=f)
            print(" ",file=f)
        
        n_words = min(10000,len(word))
        #create first plot
        fig, ax = plt.subplots(figsize=(9,7))
        plt.plot(np.arange((n_words)),count[0:n_words])
        plt.grid(axis='y')
        title = f'{n_words} most common words({when} removing stopwords and stemming)'
        plt.title(title)
        figname1 = "expl_output/" +str(n_words)+ "mostcommon"+when+".png"
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
        figname2 = "expl_output/" + str(n) + "mostcommon"+when+".png"
        plt.savefig(figname2)
        print(figname1 + " and " + figname2 + " have been updated")

    def addArticleLengthColumn(self,df):
        # adds a new column to the dataframe containing number of characters in the article
        df['articleLength'] = df['content'].apply(len)

    def run(self):
        #preprocess
        p1 = prep.preprocessor()
        df1 = p1.read_data('data/newssample.txt')
        df1Cleaned = p1.clean_data(df1)
        #make plots
        self.getMostCommon(p1.tokenized,"before")
        self.getMostCommon(df1Cleaned['content'],"after(+tail removed)")

        #preprocess without removing tail
        p2 = prep.preprocessor()
        df2 = p2.read_data('data/newssample.txt')
        #make plots
        df2Cleaned = p2.clean_data(df2,rm_tail=False)
        self.getMostCommon(df2Cleaned['content'],"after")

        self.addArticleLengthColumn(df1)

        stats_lentgh = df1['articleLength'].describe(percentiles=[0.25,0.50,0.75,0.90,0.95,0.99])
        domain_counts = df1['domain'].value_counts()
        total_domains = df1['domain'].nunique()
        column_names = df1.keys()

        with open("expl_output/exploration_stats.txt",'a') as f:
            print("Length of articles:",file=f)
            print(stats_lentgh,file=f)
            print(" ",file=f)

            print("Domains and their count:",file=f)
            print(domain_counts,file=f)
            print(" ",file=f)
            print("Total domains:",file=f)
            print(total_domains,file=f)
            print(" ",file=f)

            print("Column names:",file=f)
            print(column_names,file=f)
            print(" ",file=f)

        print("Contents in expl_output updated")

