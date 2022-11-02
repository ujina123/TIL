from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import tensorflow as tf
import gradio as gr
from PIL import Image
import numpy as np
from keras.models import load_model

examples = [
    ['study/2022_10_26/examples_img/7.png'],
    ['study/2022_10_26/examples_img/9.png'],
    ['study/2022_10_26/examples_img/1.png'],
]

model_ = load_model('study/2022_10_26/cnn_mnist1026.h5')

def greet(img):
    img_3d = img.reshape(-1, img.shape[0], img.shape[1])
    im_resize = img_3d/255.0
    prediction = model_.predict(im_resize)

    sm_layer = tf.keras.layers.Softmax()
    sm_p_score = sm_layer(prediction).numpy()
    pred = np.argmax(sm_p_score)
    return pred


im = gr.inputs.Image(shape=(28, 28),image_mode='L')
demo = gr.Interface(fn=greet, inputs=im, outputs='label',examples=examples) # or outputs=gr.outputs.Label()
demo.launch() 
