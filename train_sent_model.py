import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers,Model,metrics,optimizers,Input,regularizers,utils,losses,callbacks,Sequential

# 50 and 128 is also available
hub_embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"


# there are 2 available hyperparameters, embedding dimension, size of 1st dense layer.
def get_review_sent_model():
    # using cpu only because of tensorflow hub bug
    with tf.device('/CPU:0'):
        model=Sequential()
        model.add(hub.KerasLayer(hub_embedding,output_shape=[20],input_shape=[],dtype=tf.string,trainable=True))
        model.add(layers.Dense(16,activation='relu',name='dense1'))
        model.add(layers.Dense(1,activation='sigmoid',name='dense_final'))
    return model