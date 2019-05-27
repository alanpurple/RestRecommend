import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers,Model,metrics,optimizers,Input,regularizers,utils,losses,callbacks,Sequential

review_file='reviews.tfrecord'

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


def main():
    feature_desc={
        'text':tf.io.FixedLenFeature([],tf.string,default_value=''),
        'stars':tf.io.FixedLenFeature([],tf.float32,default_value=0.0)
        }
    def _parse_data(proto):
        parsed=tf.io.parse_single_example(proto,feature_desc)
        return (parsed['text'],parsed['stars'])

    raw_data=tf.data.TFRecordDataset([review_file])
    dataset=raw_data.map(_parse_data)
    dataset=dataset.shuffle(3000).batch(128)

    model=get_review_sent_model()

    utils.plot_model(model,'review_sent_model.png',show_shapes=True)
    # using cpu only because of tensorflow hub bug
    with tf.device('/CPU:0'):
        model.compile(optimizers.Adam(),'mae')

        model_callbacks = [
        callbacks.ModelCheckpoint(
            # cannot use whole-model saving because of hub layer
            # filepath='train/review_model_{epoch}.h5',
            filepath='train/review_model_{epoch}.ckpt',
            save_weights_only=True,
            verbose=1)
        ]

        model.fit(dataset,epochs=3,callbacks=model_callbacks)

        model.save('train/review_model_final.h5')


if __name__=='__main__':
    main()