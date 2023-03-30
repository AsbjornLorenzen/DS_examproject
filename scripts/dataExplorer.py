import preprocessing as prep
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
class data_explorer():
    def plotMostCommon(self,df,when):
        #Before cleaning
        #before - pandas series with content before removing stop words and applying stemming
        words = []
        [ words.extend(el) for el in df ]
        
        #create Counter to find count of words 
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

        #create 10.000-words-plot
        fig, ax = plt.subplots(figsize=(6,4))
        plt.plot(np.arange((n_words)),count[0:n_words])
        plt.grid(axis='y')
        title = f'{n_words} most common words({when} removing stopwords and stemming)'
        fig.subplots_adjust(left=0.25)
        plt.ylabel("Total count")

        figname1 = "expl_output/" +str(n_words)+ "mostcommon"+when+".png"
        plt.savefig(figname1)

        #create 50-words-plot
        #n is number of words to include in plot
        n = 50
        fig, ax = plt.subplots(figsize=(8,4))
        plt.bar(word[0:n],height=count[0:n])
        plt.xticks(rotation=90)
        plt.grid(axis='y')
        title = f'{n} most common words({when} removing stopwords and stemming)'
        plt.ylabel("Total count")
        if when=='after':
            plt.ylim(bottom=40)
        fig.subplots_adjust(bottom=0.25)
        figname2 = "expl_output/" + str(n) + "mostcommon"+when+".png"
        plt.savefig(figname2)
        print(figname1 + " and " + figname2 + " have been updated")

    def addArticleLengthColumn(self,df):
        # adds a new column to the dataframe containing number of characters in the article
        df['articleLength'] = df['content'].apply(len)

    def addFakeColumn(self,df):

        type_map = {
            'fake':1.0,
            'satire':1.0,
            'bias':1.0,
            'conspiracy':1.0,
            'state':1.0,
            'junksci':1.0,
            'hate':0.0,
            'clickbait':0.0,
            'unreliable':0.0,
            'political':0.0,
            'reliable':0.0
        }

        df['fake'] = df['type'].map(type_map)
        return df

    def run(self):
        #preprocess
        p1 = prep.preprocessor()
        df1 = p1.read_data('data/newssample.txt')
        #drop rows with missing labels
        df1noNan = df1.dropna(subset='type')
        df1noNan = df1.drop(df1[df1['type']=='unknown'].index)
        df1CleanedNoTail = p1.clean_data(df1noNan)
        #make plots
        self.plotMostCommon(p1.tokenized,"before")
        self.plotMostCommon(df1CleanedNoTail['content'],"after(+tail removed)")

        #preprocess without removing tail
        p2 = prep.preprocessor()
        df2 = p2.read_data('data/newssample.txt')
        df2noNan = df2.dropna(subset='type')
        df2noNan = df2.drop(df2[df2['type']=='unknown'].index)

        #make plots
        df2Cleaned = p2.clean_data(df2noNan,rm_tail=False)
        self.plotMostCommon(df2Cleaned['content'],"after")

        #add column based on current columns
        self.addArticleLengthColumn(df1)
        self.addFakeColumn(df1)

        # get stats on article lengths, domains and fraction of fake articles 
        typeCounts = df1["type"].value_counts(dropna=False)
        #remove rows where label is unknown
        df1 = df1.dropna(subset='type')
        df1 = df1.drop(df1[df1['type']=='unknown'].index)

        stats_lentgh = df1['articleLength'].describe(percentiles=[0.25,0.50,0.75,0.90,0.95,0.99])
        domain_counts = df1['domain'].value_counts()
        total_domains = df1['domain'].nunique()
        fakeMean = df1['fake'].mean()
        df1Corr = df1[['articleLength','fake']].corr()
        df1FakeAvgLength = df1.groupby('fake').agg({'articleLength':['mean','median']})


        dfdomainStats = df1[["domain","articleLength","fake"]]
        dfdomainStats['domainCount'] = 1
        dfdomainStats = dfdomainStats.groupby("domain").agg({
                                    'domainCount':'sum',
                                    'fake':'mean',
                                    'articleLength':'mean'})
        dfdomainStats = dfdomainStats.sort_values(by='domainCount',ascending=False)

        column_names = df1.keys()
        #write stats to expl_output/exploration_stats.txt
        with open("expl_output/exploration_stats.txt",'a') as f:
            print("Type counts:",file=f)
            print(typeCounts,file=f)
            print(" ",file=f)

            print("Fake mean:",file=f)
            print(fakeMean,file=f)
            print(" ",file=f)
            
            print("Length of articles:",file=f)
            print(stats_lentgh,file=f)
            print(" ",file=f)

            print("Domains and their count:",file=f)
            print(domain_counts,file=f)
            print(" ",file=f)
            print("Total domains:",file=f)
            print(total_domains,file=f)
            print(" ",file=f)

            print("Domains count, average length and fake-label:",file=f)
            print(dfdomainStats,file=f)
            print(" ",file=f)

            print("Correlation between article length and fake, for all points:",file=f)
            print(df1Corr,file=f)
            print(" ",file=f)

            print("Average and median lengths grouped by fake/true:",file=f)
            print(df1FakeAvgLength,file=f)
            print(" ",file=f)
            
            

            print("Column names:",file=f)
            print(column_names,file=f)
            print(" ",file=f)

        print("Contents in expl_output updated")

if __name__ == '__main__':
    de = data_explorer()
    de.run()

