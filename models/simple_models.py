from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import numpy as np
import pickle

class simple_models():
    def __init__(self,dataset):
        self.cv = CountVectorizer(binary=False,max_df=0.95)
        self.dataset = dataset # Needed when we need to load more than just the train/test/val csv files
    
    def logistic_model(self, train_df, val_df):
        self.model = LogisticRegression(max_iter=1500)
        self.modelname = "Logistic Regression" #used to print accuracy
        self.test_model(train_df, val_df)
    
    def linear_model(self, train_df, val_df):
        self.model = LinearRegression()
        self.modelname = "Linear Regression" #used to print accuracy and determine if we have to round predictions
        self.test_model(train_df, val_df)
    
    def dtree_model(self, train_df, val_df):
        self.model = tree.DecisionTreeClassifier()
        self.modelname = "Decision tree classifier" #used to print accuracy
        self.test_model(train_df, val_df)
    
    def passagg_model(self, train_df, val_df):
        self.model = PassiveAggressiveClassifier()
        self.modelname = "Passive aggresive classifier" #used to print accuracy
        self.test_model(train_df, val_df)

    # Splits data, fits model, and reports accuracy of predictions
    def test_model(self,train_df,val_df):
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
    
    # Makes count vectoriser or tfidf of desired field, and returns the features.
    # These features should be used to train the model.
    def get_feature_set(self,x_train,x_val,mode='cv',field='content'):
        if mode == 'cv':
             self.cv.fit_transform(x_train[field].values)
             train_feat = self.cv.transform(x_train[field].values)
             test_feat = self.cv.transform(x_val[field].values)
        elif mode == 'tfidf':
            # Does not use field arg, since the preprocessed tfidf only works with content (text)
            dir = 'data/' + self.dataset + '/'
            try:
                with open(dir + 'tfidf_vectorizer.pickle', 'rb') as handle:
                    self.tfidf = pickle.load(handle)
                with open(dir + 'tfidf_train_matrix.pickle', 'rb') as handle:
                    self.tfidf_matrix = pickle.load(handle)
            except Exception as e:
                print('Remember to create the tfidf pickle files before running with the tfidf model!\n',e)
            train_feat = self.tfidf_matrix
            test_feat = self.tfidf.transform(x_val['content'].values)
        return train_feat, test_feat

    def fit(self,x_train,y_train):
        self.model.fit(x_train,y_train)
    
    def pred(self,x_val,y_val_observed):
        y_pred = self.model.predict(x_val)
        if self.modelname == "Linear Regression": 
            y_pred = [np.round(n) for n in y_pred]
        acc = accuracy_score(y_val_observed,y_pred)
        print(self.modelname,' accuracy: ', acc)