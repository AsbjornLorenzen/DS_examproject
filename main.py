import pandas as pd
from scripts import preprocessor, preprocessor_to_text
from models import simple_models
from models import naive_bayes
from models import SVM
import gc



#df = pd.read_csv('https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv')
#print(df.keys())

class fake_news_predictor():
    def __init__(self,dataset):
        print('Initializing fake news predictors...')
        self.preprocessor = preprocessor
        self.dataset = dataset

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
        simple_models(self.dataset).logistic_model(self.train_df,self.val_df)
        print(self.train_df.shape)

    def run_linear_model(self):
        if ((not hasattr(self,'train_df')) and (not hasattr(self,'val_df'))): # Should test set also be required??
            print('Error: Dataframe was not loaded. Remember to use load_dataframes() to load at least the train and validation set')
        simple_models(self.dataset).linear_model(self.train_df,self.val_df)
    
    def run_dtree_model(self):
        if ((not hasattr(self,'train_df')) and (not hasattr(self,'val_df'))): # Should test set also be required??
            print('Error: Dataframe was not loaded. Remember to use load_dataframes() to load at least the train and validation set')
        simple_models(self.dataset).dtree_model(self.train_df, self.val_df)
    
    def run_passagg_model(self):
        if ((not hasattr(self,'train_df')) and (not hasattr(self,'val_df'))): # Should test set also be required??
            print('Error: Dataframe was not loaded. Remember to use load_dataframes() to load at least the train and validation set')
        simple_models(self.dataset).passagg_model(self.train_df, self.val_df)
    
    def run_nbayes_model(self):
        if ((not hasattr(self,'train_df')) and (not hasattr(self,'val_df'))): # Should test set also be required??
            print('Error: Dataframe was not loaded. Remember to use load_dataframes() to load at least the train and validation set')
        naive_bayes(self.dataset).naive_bayes_model(self.train_df, self.val_df)
    
    def run_SVM_model(self):
        if ((not hasattr(self,'train_df')) and (not hasattr(self,'val_df'))): # Should test set also be required??
            print('Error: Dataframe was not loaded. Remember to use load_dataframes() to load at least the train and validation set')
        SVM(self.dataset).SV_model(self.train_df, self.val_df)

    def load_dataframes(self,train_set=True,val_set=True,test_set=False):
        # Load train, val and test dataframes if they are not already loaded and if their filename is given as arg
        dir = 'data/' + self.dataset + '/'

        if ((not hasattr(self,'train_df')) and train_set):
            self.train_df = pd.read_csv(dir + 'train.csv',index_col=False,usecols=range(1,16))
            self.train_df.columns = self.column_names
            self.train_df['type'] = self.train_df['type'].map(self.type_map).fillna(1) # Sort unknown as 1 (fake)

        if ((not hasattr(self,'val_df')) and val_set):
            self.val_df = pd.read_csv(dir + 'validation.csv',index_col=False,usecols=range(1,16))
            self.val_df.columns = self.column_names
            self.val_df['type'] = self.val_df['type'].map(self.type_map).fillna(1) # Sort unknown as 1 (fake)

        if ((not hasattr(self,'test_df')) and test_set):
            self.test_df = pd.read_csv(dir + 'test.csv',index_col=False,usecols=range(1,16))
            self.test_df.columns = self.column_names
            self.test_df['type'] = self.test_df['type'].map(self.type_map).fillna(1) # Sort unknown as 1 (fake)

    # Remove dataframes from memory. Useful if we want to explicitly load a new dataset
    def remove_dataframes(self):
        if (hasattr(self,'train_df')):
            del self.train_df
        if (hasattr(self,'val_df')):
            del self.val_df
        if (hasattr(self,'test_df')):
            del self.test_df
        # Run garbage collector to erase from memory
        gc.collect()

if __name__ == '__main__':
    predictor = fake_news_predictor('apples') # 'grapes' arg is the name of the dataset (the directory) which is loaded and trained/predicted on
    predictor.load_dataframes() # load small file as training model
    predictor.run_SVM_model()
    #predictor.run_nbayes_model()
    #predictor.run_linear_model()
    #predictor.run_dtree_model()
    #predictor.run_passagg_model()