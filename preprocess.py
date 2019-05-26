import numpy as np
import pandas as pd
import tensorflow as tf

user_df=pd.read_csv('data-yelp/yelp_user.csv')
reviews_df=pd.read_csv('data-yelp/yelp_review.csv')

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

user_reduced=user_df[user_df['review_count']>4]
user_list=[elem['user_id'] for idx,elem in user_reduced.iterrows()]

reviews_df=reviews_df.drop(['review_id','business_id','date','useful','funny','cool'],axis=1)
reviews_df=reviews_df[reviews_df['stars']!=3]

data_all=[]
with tf.io.TFRecordWriter('reviews.tfrecord') as writer:
    for user,review in reviews_df.groupby('user_id'):
        if user not in user_list:
            continue
        high=False
        low=False
        review=review.drop('user_id',axis=1)
        for idx,item in review.iterrows():
            stars=item['stars']
            if not high:
                if stars==4 or stars==5:
                    data_high={'text':_bytes_feature(item.text),'stars':_float_feature((stars-1)*0.25)}
                    high=True
            if not low:
                if stars==1 or stars==2:
                    data_low={'text':_bytes_feature(item.text),'stars':_float_feature((stars-1)*0.25)}
                    low=True
            if high and low:
                break
        if high and low: # only add review data for user which has high and low both
            example_high=tf.train.Example(features=tf.train.Features(feature=data_high))
            writer.write(example_high.SerializeToString())
            example_low=tf.train.Example(features=tf.train.Features(feature=data_low))
            writer.write(example_low.SerializeToString())