import pandas as pd
from scripts import preprocessor
from models import model_logistic_regression, model_linear_regression
import gc



#df = pd.read_csv('https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv')
#print(df.keys())

class fake_news_predictor():
    def __init__(self):
        print('Initializing fake news predictor...')
        self.preprocessor = preprocessor

        # 1 means fake, 0 means true.
        self.type_map = {
            'fake':1,
            'satire':1,
            'bias':1,
            'conspiracy':1,
            'state':1,
            'junksci':1,
            'hate':0,
            'clickbait':0,
            'unreliable':0,
            'political':0,
            'reliable':0
        }
        self.column_names = [
            'id', 'domain', 'type', 'url', 'content',
            'scraped_at', 'inserted_at', 'updated_at', 'title', 'authors',
            'keywords', 'meta_keywords', 'meta_description', 'tags', 'summary']

    # Run preprocessing on n rows in the corpus. Ideally, this should only be called after the preprocessing has been modified.
    def preprocess_all_data(self,nrows,input_file='data/news_cleaned_2018_02_13.csv',output_file='data/news_cleaned_preprocessed_default'):
        self.preprocessor.bulk_preprocess(nrows,input_file,output_file)

    # Logistic model runs on df of text data, not of counter data!
    def run_logistic_model(self):
        if ((not hasattr(self,'train_df')) and (not hasattr(self,'val_df'))): # Should test set also be required??
            print('Error: Dataframe was not loaded. Remember to use load_dataframes() to load at least the train and validation set')

        self.logistic_model = model_logistic_regression()
        self.logistic_model.test_model(self.train_df,self.val_df)

    def run_linear_model(self):
        if ((not hasattr(self,'train_df')) and (not hasattr(self,'val_df'))): # Should test set also be required??
            print('Error: Dataframe was not loaded. Remember to use load_dataframes() to load at least the train and validation set')

        self.linear_model = model_linear_regression()
        self.linear_model.test_model(self.train_df,self.val_df)

    def load_dataframes(self,train_set=None,val_set=None,test_set=None):
        # Load train, val and test dataframes if they are not already loaded and if their filename is given as arg
        if ((not hasattr(self,'train_df')) and train_set):
            self.train_df = pd.read_csv(train_set,index_col=False,usecols=range(1,16))
            self.train_df.columns = self.column_names
            self.train_df['type'] = self.train_df['type'].map(self.type_map).fillna(1) # Sort unknown as 1 (fake)

        if ((not hasattr(self,'val_df')) and val_set):
            self.val_df = pd.read_csv(val_set,index_col=False,usecols=range(1,16))
            self.val_df.columns = self.column_names
            self.val_df['type'] = self.val_df['type'].map(self.type_map).fillna(1) # Sort unknown as 1 (fake)

        if ((not hasattr(self,'test_df')) and test_set):
            self.test_df = pd.read_csv(test_set,index_col=False,usecols=range(1,16))
            self.test_df.columns = self.column_names
            self.test_df['type'] = self.test_df['type'].map(self.type_map).fillna(1) # Sort unknown as 1 (fake)


    # Remove dataframes from memory. Useful if we want to explicitly load a new dataset
    def remove_dataframes(self):
        del self.train_df
        del self.val_df
        del self.test_df

        # Run garbage collector to erase from memory
        gc.collect()

import pandas as pd
from scripts import preprocessor
from models import simple_models
import gc



#df = pd.read_csv('https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv')
#print(df.keys())

class fake_news_predictor():
    def __init__(self):
        print('Initializing fake news predictors...')
        self.preprocessor = preprocessor

        # 1 means fake, 0 means true.
        self.type_map = {
            'fake':1,
            'satire':1,
            'bias':1,
            'conspiracy':1,
            'state':1,
            'junksci':1,
            'hate':0,
            'clickbait':0,
            'unreliable':0,
            'political':0,
            'reliable':0
        }
        self.column_names = [
            'id', 'domain', 'type', 'url', 'content',
            'scraped_at', 'inserted_at', 'updated_at', 'title', 'authors',
            'keywords', 'meta_keywords', 'meta_description', 'tags', 'summary']

    # Run preprocessing on n rows in the corpus. Ideally, this should only be called after the preprocessing has been modified.
    def preprocess_all_data(self,nrows,input_file='data/news_cleaned_2018_02_13.csv',output_file='data/news_cleaned_preprocessed_default'):
        self.preprocessor.bulk_preprocess(nrows,input_file,output_file)

    def run_logistic_model(self):
        if ((not hasattr(self,'train_df')) and (not hasattr(self,'val_df'))): # Should test set also be required??
            print('Error: Dataframe was not loaded. Remember to use load_dataframes() to load at least the train and validation set')
        simple_models().logistic_model(self.train_df,self.val_df)

    def run_linear_model(self):
        if ((not hasattr(self,'train_df')) and (not hasattr(self,'val_df'))): # Should test set also be required??
            print('Error: Dataframe was not loaded. Remember to use load_dataframes() to load at least the train and validation set')
        simple_models().linear_model(self.train_df,self.val_df)
    
    def run_dtree_model(self):
        if ((not hasattr(self,'train_df')) and (not hasattr(self,'val_df'))): # Should test set also be required??
            print('Error: Dataframe was not loaded. Remember to use load_dataframes() to load at least the train and validation set')
        simple_models().dtree_model(self.train_df, self.val_df)
    

if __name__ == '__main__':
    predictor = fake_news_predictor()
    smalldf = pd.read_csv('data/newssample_preprocessed.csv')
    predictor.load_dataframes('data/news_cleaned_preprocessed_text_sk_train.csv','data/news_cleaned_preprocessed_text_sk_validation.csv') # load small file as training model
    predictor.run_logistic_model()
    def run_passagg_model(self):
        if ((not hasattr(self,'train_df')) and (not hasattr(self,'val_df'))): # Should test set also be required??
            print('Error: Dataframe was not loaded. Remember to use load_dataframes() to load at least the train and validation set')
        simple_models().passagg_model(self.train_df, self.val_df)

    def load_dataframes(self,train_set=None,val_set=None,test_set=None):
        # Load train, val and test dataframes if they are not already loaded and if their filename is given as arg
        if ((not hasattr(self,'train_df')) and train_set):
            self.train_df = pd.read_csv(train_set,index_col=False,usecols=range(1,16))
            self.train_df.columns = self.column_names
            self.train_df['type'] = self.train_df['type'].map(self.type_map).fillna(1) # Sort unknown as 1 (fake)

        if ((not hasattr(self,'val_df')) and val_set):
            self.val_df = pd.read_csv(val_set,index_col=False,usecols=range(1,16))
            self.val_df.columns = self.column_names
            self.val_df['type'] = self.val_df['type'].map(self.type_map).fillna(1) # Sort unknown as 1 (fake)

        if ((not hasattr(self,'test_df')) and test_set):
            self.test_df = pd.read_csv(test_set,index_col=False,usecols=range(1,16))
            self.test_df.columns = self.column_names
            self.test_df['type'] = self.test_df['type'].map(self.type_map).fillna(1) # Sort unknown as 1 (fake)


    # Remove dataframes from memory. Useful if we want to explicitly load a new dataset
    def remove_dataframes(self):
        del self.train_df
        del self.val_df
        del self.test_df

        # Run garbage collector to erase from memory
        gc.collect()


if __name__ == '__main__':
    predictor = fake_news_predictor()
    smalldf = pd.read_csv('data/newssample_preprocessed.csv')
    predictor.load_dataframes('data/news_cleaned_preprocessed_text_train.csv','data/news_cleaned_preprocessed_text_validation.csv') # load small file as training model
    predictor.run_logistic_model()
    predictor.run_linear_model()
    predictor.run_dtree_model()
    predictor.run_passagg_model()