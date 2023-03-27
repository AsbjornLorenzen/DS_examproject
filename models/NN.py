import tensorflow as tf
#from tensorflow import keras
from keras import layers
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
    
    def use(self, train_df, val_df, test_df):
        '''fit to train_df, predict on val_df and calculate accuracy'''

        # Create file for writing model output
        f = open("model_outputs/NN_model_output.txt","w")
        f.write("File created")
        f.close()

        #split data features(x) and labels(y)
        x_train, y_train = self.split_x_y(train_df)
        x_val, y_val = self.split_x_y(val_df)
        x_test, y_test = self.split_x_y(test_df)

        #keep only 'content' column
        x_train = x_train['content']
        x_val = x_val['content']
        x_test = x_test['content']
        for x in [x_train,x_val,x_test]:
            print(f'shape of arrays(train,val,test): {x.shape}')

        #vectorize 'content' of both train, validation and test data        
        #train_feat = self.cv.fit_transform(x_train.values) #fit and transform
        #val_feat = self.cv.transform(x_val.values) #only transfor
        #test_feat = self.cv.transform(x_test.values) #only transfor

        #convert inputs to tensors(tensors are like arrays but in tensorflow instead of numpy)
        #removed since vectorize layer added
        train_x_tensor = tf.convert_to_tensor(x_train)
        train_y_tensor = tf.convert_to_tensor(y_train)
        val_x_tensor = tf.convert_to_tensor(x_val)
        val_y_tensor = tf.convert_to_tensor(y_val)
        test_x_tensor = tf.convert_to_tensor(x_test)
        test_y_tensor = tf.convert_to_tensor(y_test)

        #for x in [train_tensor,val_tensor,test_tensor]:
        #    print(f'shape of tensors(train,val,test): {x.shape}')
        train_set = tf.data.Dataset.from_tensor_slices(x_train) #prepare for vocab vectorize_layer.adapt(
        ###val_set = tf.data.Dataset.from_tensor_slices(x_val)
        ###test_set = tf.data.Dataset.from_tensor_slices(x_test)



        #find vocabulary size
        #vocab_size = train_feat[0].shape[0] 

        
        #define the model and its neural net layers (setting name to 'NNmodel')
        ##måske add conv1d, globalmaxpooling1d lag efter embedding
        vectorize_layer = layers.TextVectorization(max_tokens=5000,output_mode='int')#,standardize=None 
        vectorize_layer.adapt(train_set.batch(64))
        vocab_size = len(vectorize_layer.get_vocabulary())
        print(vocab_size)

        self.model = tf.keras.models.Sequential()
        #self.model.add(tf.keras.Input(shape=(1,), dtype=tf.string))#input layer necessary for defining shape and type of input
        self.model.add(vectorize_layer)
        self.model.add(layers.Embedding(input_dim=vocab_size, output_dim=32))
        self.model.add(layers.GlobalAveragePooling1D())
        self.model.add(layers.Dense(activation='relu',units=40)) #Relu is great when a lot of inputs are zero
        self.model.add(layers.Dense(activation='relu',units=20)) #Relu is great when a lot of inputs are zero
        self.model.add(layers.Dense(activation='relu',units=10)) #Relu is great when a lot of inputs are zero
        #self.model.add(layers.Dropout(0.05)) #used to prevent overfitting
        self.model.add(layers.Dense(activation='sigmoid',units=1))

        self.model.summary()
        #f = open("model_outputs/NN_model_output.txt","a")
        #f.write(summary) #store summary
        #f.close()

        #compile model(with stochastic gradient descent and the binary cross entropy loss function)
        self.model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

        #fit on vectorized content and adjust according to performance on validation data
        #avoid overfitting by keeping number of epochs relatively low
        self.model.fit(x=train_x_tensor, y=train_y_tensor, epochs=6, batch_size=64, validation_data=(val_x_tensor,val_y_tensor)) #adjust to 100 and 2500

        #predict on vectorized testdata data
        results = self.model.evaluate(test_x_tensor,test_y_tensor,batch_size=64)
        #f = open("model_outputs/NN_model_output.txt","a") #store results
        #f.write(results)
        #f.close()

