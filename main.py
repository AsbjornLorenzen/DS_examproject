import pandas as pd
from scripts import preprocessor


#df = pd.read_csv('https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv')
#print(df.keys())

class fake_news_predictor():
    def __init__(self):
        print('Initializing fake news predictor...')
        self.preprocessor = preprocessor

    # Run preprocessing on n rows in the corpus. Ideally, this should only be called after the preprocessing has been modified.
    def preprocess_all_data(self,nrows,input_file='data/news_cleaned_2018_02_13.csv',output_file='data/news_cleaned_preprocessed_default'):
        self.preprocessor.bulk_preprocess(nrows,input_file,output_file)

if __name__ == '__main__':
    pass