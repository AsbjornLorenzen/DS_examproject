import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import pickle 
import numpy as np
from sklearn.model_selection import GridSearchCV

class SVM():
    def __init__(self, dataset):
        self.dataset = dataset # Needed when we need to load more than just the train/test/val csv files
        self.model = SVC(kernel="linear")

    def SV_model(self, train_df, val_df):
        x_train, y_train = self.split_x_y(train_df)
        x_val, y_val = self.split_x_y(val_df)
        train_feature_set, val_feature_set = self.get_feature_set(x_train,x_val)

        self.hyp_tuning(train_feature_set, y_train)
        self.fit(train_feature_set,y_train)
        self.pred(val_feature_set,y_val)
    
    def hyp_tuning(self, train_feature_set, y_train): # Optimization of hyperparameters
        hyperparameters = {'C': [0.1, 1, 10], 'kernel': ['linear', 'poly', 'rbf'], 'gamma': [0.01, 0.1, 1]}
        # perform a grid search over the parameter grid
        grid_search = GridSearchCV(self.model, hyperparameters, cv=5)
        grid_search.fit(train_feature_set, y_train)
        # print the best hyperparameters
        print('Optimal parameters: ' + grid_search.best_params_)
        print("Training accuracy with best hyperparameters: ", grid_search.best_score_)
    
    # Makes tfidf of desired field, and returns the features.
    # These features should be used to train the model.
    def get_feature_set(self, x_train, x_val):
        dir = 'data/' + self.dataset + '/'
        try:
            with open(dir + 'tfidf_vectorizer.pickle', 'rb') as handle:
                self.tfidf = pickle.load(handle)
            with open(dir + 'tfidf_train_matrix.pickle', 'rb') as handle:
                self.tfidf_train_matrix = pickle.load(handle)
        except Exception as e:
            print('Remember to create the tfidf pickle files before running with the tfidf model!\n',e)
        train_feat = self.tfidf_train_matrix
        val_feat = self.tfidf.transform(x_val['content'].values)
        return train_feat, val_feat
    
    # Return a Y column with 1 for fake news and 0 for true
    def split_x_y(self,df):
        y = df['type'].values
        x = df.drop(['type'],axis=1)
        return x, y
    
    def fit(self,x_train,y_train):
        self.model.fit(x_train,y_train)
    
    def pred(self,x_val,y_val_observed):
        y_pred = self.model.predict(x_val)
        acc = accuracy_score(y_val_observed,y_pred)
        print('SVM accuracy: ', acc)



