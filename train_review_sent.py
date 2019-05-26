import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers,Model,metrics,optimizers,Input

review_file='reviews.tfrecord'

hub_embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"

def Review_sent():
    review_input=Input(tf.string,name='review_input')
    embedding=hub.KerasLayer(hub_embedding,input_shape=[],dtype=tf.string,trainable=True)(review_input)
    output=layers.dense(16,activation='relu',name='dense1')(embedding)
    output=layers.dense(1,activation='sigmoid',name='dense_final')(output)
    model=Model(reviereview_input,output)
    return model




def main():
    pass


if __name__=='__main__':
    main()