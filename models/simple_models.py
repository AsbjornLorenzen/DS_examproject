from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn import tree
import pandas as pd
import numpy as np
import pickle
import sklearn

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
        # Decide which sets to load: 
        # When using LIAR or another dataset, modify this line
        train_feature_set, val_feature_set = self.get_feature_set(x_train,x_val,mode='tfidf',test_set_feat='tfidf_test_matrix.pickle')
        self.fit(train_feature_set,y_train)
        self.pred(val_feature_set,y_val)

    # Return a Y column with 1 for fake news and 0 for true
    def split_x_y(self,df):
        y = df['type'].values
        x = df.drop(['type'],axis=1)
        return x, y
    
    # Makes count vectoriser or tfidf of desired field, and returns the features.
    # These features should be used to train the model.
    def get_feature_set(self,x_train,x_val,mode='cv',field='content',test_set_feat=''):
        if mode == 'cv':
             # We don't count vectorize in preprocessing, so we do it here:
             self.cv.fit_transform(x_train[field].values)
             train_feat = self.cv.transform(x_train[field].values)
             test_feat = self.cv.transform(x_val[field].values)

        elif mode == 'tfidf':
            # Does not use field arg, since the preprocessed tfidf only works with content (text)
            # We apply tfidf in preprocessing, so load preprocessed train matrix:
            dir = 'data/' + self.dataset + '/'
            try:
                with open(dir + 'tfidf_vectorizer.pickle', 'rb') as handle:
                    self.tfidf = pickle.load(handle)
                with open(dir + 'tfidf_train_matrix.pickle', 'rb') as handle:
                    train_feat = pickle.load(handle)
            except Exception as e:
                print('Remember to create the tfidf pickle files before running with the tfidf model!\n',e)
            
            # If we already have a transformed test set matrix:
            if test_set_feat:
                with open(dir + test_set_feat, 'rb') as handle:
                    test_feat = pickle.load(handle)
            else:
                test_feat = self.tfidf.transform(x_val['content'].values)

        # Word2vec and tfidf are always applied in preprocessing
        elif mode == 'word2vec_tfidf':
            dir = 'data/' + self.dataset + '/'
            try:
                with open(dir + 'word2vec_tfidf_train.pickle', 'rb') as handle:
                    self.train_feat = pickle.load(handle)
                with open(dir + 'word2vec_tfidf_val.pickle.pickle', 'rb') as handle:
                    self.test_feat = pickle.load(handle)
            except Exception as e:
                print('Remember to create the tfidf pickle files before running with the tfidf model!\n',e)
        return train_feat, test_feat

    def fit(self,x_train,y_train):
        self.model.fit(x_train,y_train)
    
    def pred(self,x_val,y_val_observed):
        y_pred = self.model.predict(x_val)

        if self.modelname == "Linear Regression": 
            y_pred = [np.round(n) for n in y_pred]
            precision = round(precision_score(y_val_observed, y_pred,average='micro'), 3)
            recall = round(recall_score(y_val_observed, y_pred,average='micro'), 3)
            f1 = round(f1_score(y_val_observed, y_pred,average='micro'), 3)
        else:
            precision = round(precision_score(y_val_observed, y_pred), 3)
            recall = round(recall_score(y_val_observed, y_pred), 3)
            f1 = round(f1_score(y_val_observed, y_pred), 3)
        
        accuracy = round(accuracy_score(y_val_observed, y_pred), 3)
        confusion_matrix = sklearn.metrics.confusion_matrix(y_val_observed,y_pred)

        print('\n\n ------------------ \n\n')
        print(self.modelname,"Accuracy: ", accuracy)
        print(self.modelname,"Precision: ", precision)
        print(self.modelname,"Recall: ", recall)
        print(self.modelname,"F1 score: ", f1)
        print(self.modelname,'Confusion Matrix: \n',confusion_matrix)
        print('\n\n ------------------ \n\n')