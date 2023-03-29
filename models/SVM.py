import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import pickle 
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import loguniform
from cleantext.sklearn import CleanTransformer

class SVM():
    def __init__(self, dataset):
        self.dataset = dataset # Needed when we need to load more than just the train/test/val csv files
        self.model = SVC(kernel='rbf',verbose=True)

    def SV_model(self, train_df, val_df):
        x_train, y_train = self.split_x_y(train_df)
        x_val, y_val = self.split_x_y(val_df)
        #train_feature_set, val_feature_set = self.get_feature_set_word2vec(x_train,x_val)
        #train_feature_set, val_feature_set = self.get_feature_set(x_train,x_val)
        with open('data/' + self.dataset + '/word2vec_train.pickle', 'rb') as handle:
            train_feature_set = pickle.load(handle)
        with open('data/' + self.dataset + '/word2vec_val.pickle', 'rb') as handle:
            val_feature_set = pickle.load(handle)
        #self.hyp_tuning(train_feature_set, y_train)
        #self.randomized_search_tuning(train_feature_set,y_train)
        self.fit(train_feature_set,y_train)
        self.pred(val_feature_set,y_val)
    
    def hyp_tuning(self, train_feature_set, y_train): # Optimization of hyperparameters
        print("Beginning hyperparameter optimization...")
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

    def randomized_search_tuning(self, train_feature_set, y_train):
        print("Beginning randomized search optimization...")
        params = {'C': loguniform(1e-6, 1e+6),
          'gamma': loguniform(1e-6, 1e+1),
          'kernel': ['linear', 'rbf']}
        search = RandomizedSearchCV(self.model, params, n_iter=25, cv=5, n_jobs=-1, scoring='accuracy', verbose=1, random_state=42)
        search.fit(train_feature_set, y_train)
        print('Best hyperparameters:', search.best_params_)
        print('Best accuracy score:', search.best_score_)
    
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
    
    """     # Makes tfidf of desired field, and returns the features.
    # These features should be used to train the model.
    def get_feature_set_word2vec(self, x_train, x_val):
        cleaner = CleanTransformer(
            # Modified from clean-texts site:
            lower=True,                    # lowercase text
            no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
            no_urls=True,                  # replace all URLs with a special token
            no_emails=True,                # replace all email addresses with a special token
            no_numbers=True,               # replace all numbers with a special token
            no_currency_symbols=False,
            replace_with_url="URLtoken",
            replace_with_email="EMAILtoken",
            replace_with_number="",
            replace_with_currency_symbol="CURtoken",
            no_punct=True,
            lang="en")
        dir = 'data/' + self.dataset + '/'
        try:
            with open(dir + 'tfidf_vectorizer.pickle', 'rb') as handle:
                self.tfidf = pickle.load(handle)
            with open(dir + 'word2vec_model.pickle', 'rb') as handle:
                self.word2vec_model = pickle.load(handle)
            with open(dir + 'word2vec_combined.pickle', 'rb') as handle:
                self.word2vec_matrix = pickle.load(handle)
        except Exception as e:
            print('Remember to create the tfidf pickle files before running with the tfidf model!\n',e)
        train_feat = self.word2vec_matrix
        tokenizer = lambda docs: [self.tfidf.build_tokenizer()(doc) for doc in docs]
        #Create feat set for val:
        #texts_val = x_val['content'].values
        texts_val = cleaner.transform(x_val['content'])
        print('TEXTS VAL ',texts_val[0:2])
        tokens_val = tokenizer(texts_val)

        print('TOKEN VAL ',tokens_val[0:2])
        sentences_val = tokens_val #[text.split() for text in texts_val]
        sentences_val_2 = []
        print('SENT VAL ',sentences_val[0:2])
        test = sentences_val[0]
        for sent in sentences_val:
            append = [word for word in sent if word in self.word2vec_model.wv]
            sentences_val_2.append(append)
        sentences_val = sentences_val_2
        print('SENT VAL TEST',sentences_val[0],test)
        #print('SENTENCES: ',sentences_val[0:10])
        print(self.word2vec_model.wv.most_similar('washington'))

        print(len(texts_val),len(sentences_val))

        print('SEnt before  ',sentences_val[0:2])
        sentences_as_text = []
        for sent in sentences_val:
            sentences_as_text.append(" ".join(sent))
        print('SEnt after  ',sentences_as_text[0:2])

        tfidf_vectors = self.tfidf.transform(sentences_as_text) # should maybe be texts_val
        #combined_vectors_val = np.zeros((len(texts_val), tfidf_vectors.shape[1] + self.word2vec_model.vector_size))
        combined_vectors_val = np.zeros((len(texts_val), self.word2vec_model.vector_size))


        #print(len(tfidf_vectors),len(sentences_val))
        for i in range(len(texts_val)):
            #print('SENT i: ',sentences_val[i])
            #allowed_words = [word for word in sentences_val[i] if word in self.word2vec_model.wv]
            #for word in sentences_val[i]:
                #allowed_words =
            #tfidf_vector = tfidf_vectors[i].toarray().ravel()
            #word2vec_vector = self.word2vec_model.wv[sentences_val[i]].mean(axis=0)
            print('sent val ',sentences_val[i])
            combined_vectors_val[i] = self.word2vec_model.wv[sentences_val[i]].mean(axis=0) #np.concatenate((tfidf_vector, word2vec_vector))
        val_feat = combined_vectors_val

        return train_feat, val_feat """
    
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



