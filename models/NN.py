import tensorflow as tf
#from tensorflow import keras
from keras import layers, metrics
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from models import simple_models
import pickle

class NN_model():
    '''Input must be cleaned, and not vectorized text'''
    def __init__(self,dataset):
        self.cv = CountVectorizer(binary=False,max_df=0.95)
        self.dataset = dataset # Needed when we need to load more than just the train/test/val csv files
    
    def split_x_y(self,df):
        y = df['type'].values
        x = df.drop(['type'],axis=1)
        return x, y
    
    def use(self, train_df, val_df,test_df,model_name='standard',use_saved_model=False):
        '''
        fit to train_df, predict on val_df and calculate accuracy
        model must be 'standard', 'LSTM' or 'word2vec'
        '''
        valid_models=['standard','LSTM','word2vec']
        if model_name not in valid_models:
            raise ValueError("model not valid")

        #split data features(x) and labels(y)
        x_train, y_train = self.split_x_y(train_df)
        x_val, y_val = self.split_x_y(val_df)
        x_test, y_test = self.split_x_y(test_df)

        #keep only 'content' column
        x_train = x_train['content']
        x_val = x_val['content']
        x_test = x_test['content']
        for x in [x_train,x_val]:
            print(f'shape of arrays(train,val/test): {x.shape}')

        #convert inputs to tensors(tensors are like arrays but in tensorflow instead of numpy)
        #removed since vectorize layer added
        train_x_tensor = tf.convert_to_tensor(x_train)
        train_y_tensor = tf.convert_to_tensor(y_train)
        val_x_tensor = tf.convert_to_tensor(x_val)
        val_y_tensor = tf.convert_to_tensor(y_val)
        test_x_tensor = tf.convert_to_tensor(x_test)
        test_y_tensor = tf.convert_to_tensor(y_test)
        if use_saved_model:
            model = tf.keras.models.load_model('NN_saved_model/'+model_name)
            model.summary()
            model.evaluate(test_x_tensor,test_y_tensor,batch_size=64)
            pred = model.predict(test_x_tensor,batch_size=64).round()
            print(tf.math.confusion_matrix(test_y_tensor,pred,num_classes=2))

            #get LIAR and evaluate:
            liar_df = pd.read_csv('data/liar/liar-preprocessed.csv')
            liar_x = tf.convert_to_tensor(liar_df['content'])
            liar_y = tf.convert_to_tensor(liar_df['type'])
            model.evaluate(liar_x,liar_y,batch_size=64)
            liar_pred = model.predict(liar_x,batch_size=64).round()
            print(tf.math.confusion_matrix(liar_y,liar_pred,num_classes=2))

            #end function call
            return

        print(tf.math.reduce_mean(train_y_tensor))
        print(tf.math.reduce_mean(val_y_tensor))
        #for x in [train_tensor,val_tensor,test_tensor]:
        #    print(f'shape of tensors(train,val,test): {x.shape}')
        train_set = tf.data.Dataset.from_tensor_slices(x_train) #prepxare for vocab vectorize_layer.adapt(
        ###val_set = tf.data.Dataset.from_tensor_slices(x_val)
        ###test_set = tf.data.Dataset.from_tensor_slices(x_test)
        
        if model_name!='word2vec':
        #train vectorization
            vectorize_layer = layers.TextVectorization(max_tokens=5000,output_mode='int',ngrams=None,standardize="lower_and_strip_punctuation")
            vectorize_layer.adapt(train_set.batch(64))
            vocab_size = len(vectorize_layer.get_vocabulary())
        
        #define the model and its neural net layers
        self.model = tf.keras.models.Sequential()
        
        if model_name!='word2vec':
            self.model.add(tf.keras.Input(shape=(1,), dtype=tf.string))#input layer necessary for defining shape and type of input
            self.model.add(vectorize_layer)
            self.model.add(layers.Embedding(input_dim=vocab_size, output_dim=32))
                
        if model_name=='standard':
            #layers
            self.model.add(layers.GlobalAveragePooling1D()) 
            #self.model.add(layers.Dense(activation='relu',units=40))
            self.model.add(layers.Dense(activation='relu',units=20))  #Relu is great when a lot of inputs are zero
            self.model.add(layers.Dense(activation='relu',units=5)) 
            #self.model.add(layers.Dropout(0.10)) #used to prevent overfitting
            self.model.add(layers.Dense(activation='sigmoid',units=1))

        if model_name == 'word2vec':
            #load word2vec pickles and convert to tensors for model input
            with open('data/' + self.dataset + '/word2vec_train.pickle', 'rb') as handle:
                pickle_train = pickle.load(handle)
                print(pickle_train.shape)
                train_x_tensor = tf.convert_to_tensor(pickle_train) #change the input to the model
                
            with open('data/' + self.dataset + '/word2vec_val.pickle', 'rb') as handle:
                pickle_val = pickle.load(handle)
                val_x_tensor = tf.convert_to_tensor(pickle_val)
            self.model.add(tf.keras.Input(shape=(300,)))#dtype=np.dtype, input layer necessary for defining shape and type of input
            #self.model.add(layers.Embedding(input_dim=vocab_size, output_dim=32))
            self.model.add(layers.Dense(activation='relu',units=20))
            self.model.add(layers.Dense(activation='relu',units=5))
            self.model.add(layers.Dense(activation='sigmoid',units=1))

        if model_name=='LSTM':
            #layers
            self.model.add(layers.LSTM(64,activation='relu',max_length=5000))
            self.model.add(layers.Dense(activation='relu',units=10)) #Relu is great when a lot of inputs are zero
            self.model.add(layers.Dense(activation='sigmoid',units=1))

        self.model.summary()

        #compile model(with adam and the binary cross entropy loss function)
        self.model.compile(optimizer='adam',loss='binary_crossentropy',metrics=[metrics.BinaryAccuracy(),metrics.Precision(),metrics.Recall()])

        #fit on vectorized content and adjust according to performance on validation data
        #avoid overfitting by keeping number of epochs relatively low
        self.model.fit(x=train_x_tensor, y=train_y_tensor, epochs=6, batch_size=64, validation_data=(val_x_tensor,val_y_tensor)) #adjust to 100 and 2500
        
        self.model.save('NN_saved_model/'+model_name)

        #predict on vectorized testdata data
        self.model.evaluate(test_x_tensor,test_y_tensor,batch_size=64)
        pred = self.model.predict(test_x_tensor,batch_size=64)
        print(tf.math.confusion_matrix(test_y_tensor,pred))


