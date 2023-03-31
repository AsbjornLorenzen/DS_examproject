# Data Science final project
To run the project, excecute main.py.
Below, you will find a documentation of the code, and details on dependencies are found at the bottom.

# Preprocessing

### preprocessor_to_text()
Our preprocessing class, which holds all methods that are used for preprocessing. It has many helper functions for handling various steps in preprocessing, but the functions that should be exported for later use are:

### clean_data
```
Params:
df - the dataframe which is cleaned. 
```

The text in the dataframe's 'content' column gets tokenized, has stopwords removed, and is stemmed.
Returns pandas dataframe with preprocessing applied to 'content' field.

### bulk_preprocess
```
Params:
nrows - number of rows in input file that are preprocessed
input_file: Filename of input file (e.g. the corpus)
output_dir: Directory of preprocessed datasets
```

Splits data in train/test/validation, and applies preprocessing to each. Data is loaded using chunks, so arbitrarily large files can be handled. Data is also saved to the resulting train/test/validation .csv files, so we never need large memory to handle even huge input files.

### bulk_preprocess_tfidf
```
Params:
nrows - number of rows in input file that are preprocessed
input_file: Filename of input file (e.g. the corpus)
output_name: Directory name of preprocessed datasets
```

Splits data in train/test/validation using chunking. Then, TF-IDF is applied using sklearn's TfidfVectorizer, which also preprocesses the data (using our own stopwords list and our own tokenizer function, so that the preprocessing is identical to that which is done using the other preprocessing methods). The TfidfVectorizer is fitted only on the train data as to avoid data leakage, and this fitted vectoriser is then used to transform the test and validation sets to feature sets (TF-IDF matrixes).
Note: Any data in the directory is wiped when calling this method. The purpose is to build a complete train/test/validation set from a corpus or a subset of the corpus.

### reservoir_sample
```Params:
n: Number of articles to sample
```

An efficient way to exctract many random articles spread throughout the whole dataset. We load data using chunks, and use a random number generator to decide whether to include every article in our output set. The likelihood of including each article is n / totn, where totn is the total number of articles in the corpus. 
This implementation also removes any rows that are missing data in the type or content fields. It uses chunking for loading, so large data files can be handled. It is also quite efficient - selecting a million articles from the 30GB corpus takes around 10 minutes on an average laptop.

Details: See https://en.wikipedia.org/wiki/Reservoir_sampling

### word2vec_preprocessing
```Params:
input_dir: Directory of input files (e.g. the corpus)
output_dir Directory to output the word2vec model and the word2vec vectors. Usually, this should be the same as the input dir.
```
This method requires a train/test/val split of the data in the input directory.
Reads train/test/validation sets from the directory, cleans the data, and preprocesses using word2vec. The word2vec model is fitted on the training data, and is then applied to the validation and/or test data set. The word2vec model includes any words that appear at least 5 times in the whole training set, and the word2vec vectors are in 300 dimensions. After fitting, we remove any words that are not in the vocabulary of word2vec, and then calculate the word vectors. For each article, we take the average of all the word vectors, and use this average to describe the whole document. The resulting word2vec matrixes are stored as pickles, which can easily be loaded by our models.

### word2vec_tfidf_preprocessing
```
Params:
input_dir: Directory of input files (e.g. the corpus)
output_dir Directory to output the word2vec model and the word2vec vectors. Usually, this should be the same as the input dir. 
```
This method requires a train/test/val split of the data in the input directory.
Similar to the two above, we load the datasets and fit tfidf and word2vec to the training data before transforming the validation/test sets. We then concatenate the word2vec and tfidf vectors in each document, and save the resulting feature set as a pickle. Note that this is a lot of data for each document, so it can be expected that the pickle output is around twice the size of the input .csv data. Therefore, this preprocessing limits the amount of articles that can be handled by a given machine.

### load_liar
```
Params:
filename (str): Name of the liar data file. 
output_dir (str): Directory where the data is written.
preprocess (bool): Determines whether the data is preprocessed with tfidf and word2vec.
```

Used for loading and modifying the liar set so it can be passed as input to any of our models. We start off by mapping the article types in liar to our decided true/false map. We have decided on:
```
            'pants-fire':1, 
            'false':1, 
            'barely-true':1,
            'half-true':1,
            'mostly-true':0,
            'true':0
```
As in the fake news corpus, we have chosen to include mostly true news sources as true/reliable news. 
We then create a new dataframe with only two columns, 'type' and 'content', which is filled with the relevant data from liar. This is saved as an unprocessed csv file that can be used for testing/exploration. 
If the flag preprocess is true, we then apply our preprocessing pipeline. We tokenize, remove stopwords, and stem, and then create a pickle for all 3 preprocessing types that we've used for the corpus: tf-idf, word2vec, and tf-idf+word2vec.

# Running models
### fake_news_predictor
Models are trained and tested from the fake_news_predictor in main.py. Before testing models, the dataset is loaded into the fake_news_predictor, and after that, any number of models can be run sequentially.

### load_dataframes
```
Params:
train_set(str): Name of the training set, defaults as 'train.csv'.
val_set(str): Name of validation set, defaults as 'validation.csv'.
test_set(str): Name of test set, defaults as an empty string, meaning that the test set is not loaded. The test set should only be loading when we are ready to evaluate the model, after tuning hyperparameters to the validation set.
liar(bool): Whether the liar set is loaded
```
The input .csv files are loaded as pandas dataframes, and the column names are set. The function returns nothing, but stores the dataframes as attributes to fake_news_predictor.

### remove_dataframes
Useful for deleting dataframes from the class instance, if we want to try loading another dataset while the program is still running.

### run_models
Runs all our models using the dataset loaded previously. We have a method for each model, and each of them work the same way. The methods for loading are:
```
    run_logistic_model()
    run_linear_model()
    run_dtree_model()
    run_passagg_model()
    run_nbayes_model()
    run_NN_model()
    run_SVM_model()
```

# Advanced Models
Our advanced models have more features, and these will briefly be described here. They can still be run as described above, but if you wish to tune them or modify them, this section is for you:

## SVM
```
Params:
dataset(str): The name of the dataset directory which we load/write data in.
```
The support vector machine class takes a dataset as input, and creates a SVC model using sklearn's implementation.

### SV_model
```
Params:
train_df(dataframe): The dataframe which will be trained on
val_df(dataframe): The dataframe which will be validated on. 
mode(str: 'tfidf' | 'word2vec' | 'word2vec_tfidf'): Which preprocessing input should be used.
kernel_approximation(bool): Whether kernel approximation is used. 
```
This is the most important method of the class, and is used for running the model.
The class instance loads the feature sets based on the input, fits to the input, and predicts based on the validation set.
Using kernel approximation speeds up training dramatically, but also reduces the quality of the model.

### hyp_tuning
```
Params:
train_df(dataframe): The dataframe which will be trained on
val_df(dataframe): The dataframe which will be validated on. 
mode(str: 'tfidf' | 'word2vec' | 'word2vec_tfidf'): Which preprocessing input should be used.
```
Used for hyperparameter tuning. The input is loaded as in SV_model, and we then apply grid search to find the best hyperparameters. This takes a long time, and it is recommended to run it on small datasets. The best hyperparameters and resulting training accuracy will be printed after the grid search has finished.

### randomized_search_tuning
```
Params:
train_df(dataframe): The dataframe which will be trained on
val_df(dataframe): The dataframe which will be validated on. 
mode(str: 'tfidf' | 'word2vec' | 'word2vec_tfidf'): Which preprocessing input should be used.
```
Similar to hyp_tuning, but with randomized search instead of grid search.

### test_on_liar
```
Params:
model_name(str): The name of the previously trained model, which should be stored as a pickle in the current directory. 
```
Requires a preprocessed liar set in the directory (see how this is achieved in the preprocessing section). Also requires a trained SVM model. This model is loaded, and predicts on the liar set. The results are written in the terminal.

## Neural Network
To run the neural network with the saved trained model, first load dataframe including the test_set with ```predictor.load_dataframes(test_set='test.csv')```
Then call run_NN_model with model_name set to 'standard' and use_saved_model = True.


# Dependencies

## Conda
Packages should be handled with conda rather than pip. Packages are found and installed using:

```conda search numpy ``` (see if numpy exists)


```conda install numpy```

When new packages are installed, remember to update the requirements and environment files as such:

```conda list > requirements.txt```

```conda env export > environment.yml```


## Setup environment
Install dependencies by entering:

```conda env create -f environment.yml```

Or with pip:

 ```pip install -r requirements.txt```


