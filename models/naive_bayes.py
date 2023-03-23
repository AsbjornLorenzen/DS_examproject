from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import numpy as np

class naive_bayes():
    def naive_bayes_model(train_feature_set, y_train, val_feature_set, y_val):
        self.model = MultinomialNB()
        self.fit(train_feature_set,y_train)
        self.pred(val_feature_set,y_val)
    
    def fit(self,x_train,y_train):
        self.model.fit(x_train,y_train)
    
    def pred(self,x_val,y_val_observed):
        y_pred = self.model.predict(x_val)
        acc = accuracy_score(y_val_observed,y_pred)
        print('Naive Bayes accuracy: ', acc)