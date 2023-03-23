from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import numpy as np
import pickle

class model_logistic_regression():
    def __init__(self):
        self.model = LogisticRegression(max_iter=1500)
        self.cv = CountVectorizer(binary=False,max_df=0.95)

    def get_df_info(self,df):
        print(f"Dataframe shape: {df.shape}")
        print(f"Dataframe columns: {df.columns} \n Df row: {df.loc[[1]]}")

    # Splits data, fits model, and reports accuracy of predictions
    def test_model(self,train_df,val_df):
        self.get_df_info(train_df)
        x_train, y_train = self.split_x_y(train_df)
        x_val, y_val = self.split_x_y(val_df)
        
        train_feature_set, val_feature_set = self.get_feature_set(x_train,x_val,mode='tfidf')

        self.fit(train_feature_set,y_train)
        self.pred(val_feature_set,y_val)


    # Return a Y column with 1 for fake news and 0 for true
    def split_x_y(self,df):
        y = df['type'].values
        x = df.drop(['type'],axis=1)
        return x, y
    
    # Makes count vectoriser of desired field, and returns the features.
    # These features should be used to train the model.
    def get_feature_set(self,x_train,x_test,mode='cv',field='content'):
        if mode == 'cv':
            self.cv.fit_transform(x_train[field].values)
            train_feat = self.cv.transform(x_train[field].values)
            test_feat = self.cv.transform(x_test[field].values)
        elif mode == 'tfidf':
            # Does not use field arg, since the preprocessed tfidf only works with content (text)
            try:
                with open('data/tfidf_vectorizer.pickle', 'rb') as handle:
                    self.tfidf = pickle.load(handle)
                with open('data/tfidf_matrix.pickle', 'rb') as handle:
                    self.tfidf_matrix = pickle.load(handle)
            except Exception as e:
                print('Remember to create the tfidf pickle files before running with the tfidf mode!\n',e)
            train_feat = self.tfidf_matrix
            test_feat = self.tfidf.transform(x_test['content'].values)

        return train_feat, test_feat


    def fit(self,x_train,y_train):
        self.model.fit(x_train,y_train)
    
    def pred(self,x_val,y_val_observed):
        y_pred = self.model.predict(x_val)
        #For linear model:
        #y_pred = [np.round(n) for n in y_pred]
        acc = accuracy_score(y_val_observed,y_pred)
        print(f"Length: {len(y_val_observed),len(y_pred)}")
        print(sum(y_val_observed),sum(y_pred))
        print('Predicted with accuracy: ',acc)

    