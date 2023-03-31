import pandas as pd
from sklearn.svm import SVC
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
import pickle 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import loguniform
from cleantext.sklearn import CleanTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import sklearn

class SVM():
    def __init__(self, dataset):
        self.dataset = dataset # Needed when we need to load more than just the train/test/val csv files
        self.model = SVC(kernel='rbf',verbose=True,cache_size=1000,max_iter=100000)

    def load_data(self,train_df,val_df,mode):
        x_train, y_train = self.split_x_y(train_df)
        x_val, y_val = self.split_x_y(val_df)

        # Get feature sets by loading pickles. The mode determines what pickle names are loaded.
        # Can load either tfidf, word2vec, or tfidf+word2vec pickles. Note that these are pre-generated in the preprocessor
        train_feature_set, val_feature_set = self.get_feature_set(x_train,x_val,mode=mode)
        return x_train, y_train, x_val, y_val, train_feature_set, val_feature_set

    def SV_model(self, train_df, val_df,mode='tfidf',kernel_approximation=False):
        x_train, y_train, x_val, y_val, train_feature_set, val_feature_set = self.load_data(train_df,val_df,mode)

        # Using kernel approximation:
        if kernel_approximation:
            n_components = 100
            approx_kernel = Nystroem(kernel='rbf',n_components=n_components, random_state=42)
            self.model = make_pipeline(approx_kernel,self.model,verbose=True)

        self.fit(train_feature_set,y_train)
        # Save model: 
        with open(self.dataset +  'trained_svm_model_tfidf_w2v_100k.pickle', 'wb') as handle:
            pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.pred(val_feature_set,y_val)
    
    def hyp_tuning(self,train_df,val_df,mode='tfidf'): # Optimization of hyperparameters
        print("Beginning hyperparameter optimization...")
        _, y_train, _, _, train_feature_set, _ = self.load_data(train_df,val_df,mode)
        hyperparameters = {
            'C': [0.1, 1, 10, 100],
            #'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4],
            'gamma': ['scale', 'auto'] + [0.1, 1, 10],
            'coef0': [-1, 0, 1]
        }

        # perform a grid search over the parameter grid
        grid_search = GridSearchCV(self.model, hyperparameters, cv=5, n_jobs=-1, verbose=1, scoring='accuracy')
        grid_search.fit(train_feature_set, y_train)

        # print the best hyperparameters
        print("Best hyperparameters: ", grid_search.best_params_)
        print("Training accuracy with best hyperparameters: ", grid_search.best_score_)


    def randomized_search_tuning(self,train_df,val_df,mode='tfidf'): # Optimization of hyperparameters
        _, y_train, _, _, train_feature_set, _ = self.load_data(train_df,val_df,mode)
        print("Beginning randomized search optimization...")
        params = {'C': loguniform(1e-6, 1e+6),
          'gamma': loguniform(1e-6, 1e+1),
          'kernel': ['linear', 'rbf']}
        search = RandomizedSearchCV(self.model, params, n_iter=25, cv=5, n_jobs=-1, scoring='accuracy', verbose=1, random_state=42)
        search.fit(train_feature_set, y_train)
        print('Best hyperparameters:', search.best_params_)
        print('Best accuracy score:', search.best_score_)

    def test_on_liar(self,model_name):
        with open('data/' + self.dataset + model_name, 'rb') as handle:
            self.model = pickle.load(handle)
        with open('data/' + self.dataset + '/tfidf_liar_matrix.pickle', 'rb') as handle:
            self.liar_feat = pickle.load(handle)
        liar_df = pd.read_csv('data/' + self.dataset + '/liar.csv',index_col=False)#,usecols=range(1,16))
        x_liar, y_liar = self.split_x_y(liar_df)
        self.pred(self.liar_feat,y_liar)
    
    # Makes tfidf of desired field, and returns the features.
    # These features should be used to train the model.
    # val_set_feat is a pre-generated feature set for validation
    def get_feature_set(self, x_train, x_val,mode='tfidf',val_set_feat=''):
        dir = self.dataset 
        if mode == 'tfidf':
            with open(dir + 'tfidf_train_matrix.pickle', 'rb') as handle:
                train_feat = pickle.load(handle)

            # If we already have a transformed test set matrix:
            if val_set_feat:
                with open(dir + val_set_feat, 'rb') as handle:
                    val_feat = pickle.load(handle)
            
            # Otherwise, generate val feat set
            else:
                with open(dir + 'tfidf_vectorizer.pickle', 'rb') as handle:
                    tfidf = pickle.load(handle)
                val_feat = tfidf.transform(x_val['content'].values)
        
        elif mode == 'word2vec':
            with open('data/' + self.dataset + '/word2vec_train.pickle', 'rb') as handle:
                train_feat = pickle.load(handle)
            with open('data/' + self.dataset + '/word2vec_val.pickle', 'rb') as handle:
                val_feat = pickle.load(handle)

        elif mode == 'word2vec_tfidf':
            with open('data/' + self.dataset + '/word2vec_tfidf_train.pickle', 'rb') as handle:
                train_feat = pickle.load(handle)
            with open('data/' + self.dataset + '/word2vec_tfidf_val.pickle', 'rb') as handle:
                val_feat = pickle.load(handle)
    
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

        precision = round(precision_score(y_val_observed, y_pred), 3)
        recall = round(recall_score(y_val_observed, y_pred), 3)
        f1 = round(f1_score(y_val_observed, y_pred), 3)
        accuracy = round(accuracy_score(y_val_observed, y_pred), 3)
        confusion_matrix = sklearn.metrics.confusion_matrix(y_val_observed,y_pred)

        print('\n\n ------------------ \n\n')
        print("Accuracy: ", accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 score: ", f1)
        print('Confusion Matrix: \n',confusion_matrix)
        print('\n\n ------------------ \n\n')



