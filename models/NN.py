import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from models import simple_models
import pickle

class NN_model():
    def __init__(self,dataset):
        self.cv = CountVectorizer(binary=False,max_df=0.95)
        self.dataset = dataset # Needed when we need to load more than just the train/test/val csv files
    
    def split_x_y(self,df):
        y = df['type'].values
        x = df.drop(['type'],axis=1)
        return x, y
    
    def use(self, train_df, val_df):
        '''fit to train_df, predict on val_df and calculate accuracy'''

        #split data features(x) and labels(y)
        x_train, y_train = self.split(train_df)
        x_val, y_val = self.split(val_df)

        #keep only 'content' column
        x_train = x_train['content']
        y_train = y_train['content']

        #vectorize 'content' of both train and test data        
        train_feat = self.cv.fit_transform(x_train.values)
        test_feat = self.cv.transform(x_val.values)

        #convert inputs to tensors(tensors are like arrays but in tensorflow instead of numpy)
        train_tensor = tf.convert_to_tensor(train_feat)
        test_tensor = tf.convert_to_tensor(test_feat)

        #find vocabulary size
        vocab_size = train_feat[0].shape[0]

        #define the model and its neural net layers (setting name to 'NNmodel')
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=vocab_size,output_dim=64),
            tf.keras.layers.Dense(activation='Relu'), #Relu is great when a lot of inputs are zero
            tf.keras.layers.Dropout(0.15), #used to prevent overfitting
            tf.keras.layers.Dense(activation='Relu'),
        ],"NNmodel")

        #compile model(with stochastic gradient descent and the binary cross entropy loss function)
        self.model.compile(optimizer='SGD',loss='binary_crossentropy',metrics=['accuracy'])

        #fit on vectorized content
        #avoid overfitting by keeping number of epochs relatively low
        self.model.fit(x=train_feat, y=y_train,epochs=3,batch_size=5000) #adjust to 100 and 2500

        #predict on vectorized test data
        

        #calculate accuracy
        




    #måske add conv1d, globalmaxpooling1d lag efter embedding