import pandas as pd
from scripts import preprocessor
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split


#df = pd.read_csv('https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv')
#print(df.keys())

def prepare_data():
    print('Preparing data...')
    starttime = timer()
    p = preprocessor()

    loaded_chunks = 0
    chunksize = 5000
    output_file = 'data/news_cleaned_preprocessed_2'

    for chunk in pd.read_csv('data/news_cleaned_2018_02_13.csv', chunksize=chunksize,nrows=500000,engine='python'):
        processed_chunk = p.clean_data(chunk,verbosity=1)

        train, remaining = train_test_split(
            processed_chunk,test_size=0.2,random_state=42
        )

        validation, test = train_test_split(
            remaining,test_size=0.5,random_state=42
        )

        train.to_csv(output_file+'_train.csv', mode='a', index=False, header=False)
        validation.to_csv(output_file+'_validation.csv', mode='a', index=False, header=False)
        test.to_csv(output_file+'_test.csv', mode='a', index=False, header=False)

        loaded_chunks += chunksize
        endtime = timer()
        print(f"Done loading {loaded_chunks} chunks in {round(endtime-starttime,3)} seconds")

if __name__ == '__main__':
    prepare_data()