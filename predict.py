import tensorflow as tf
from review_sent_model import get_review_sent_model,hub_embedding
import numpy as np


model=get_review_sent_model()

model.load_weights('train/review_model_3.ckpt')

pred=model.predict(np.array(['hahaha, this place is nonsense.','smells like shit','how wonderful!']))

print(pred)