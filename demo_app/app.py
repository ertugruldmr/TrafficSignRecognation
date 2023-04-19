import gradio as gr
import tensorflow as tf
from glob import glob
import numpy as np
import pickle
import json

# loading the model
model_path = "custom_conv_model"
model = tf.keras.models.load_model(model_path)

# loading the example images to test
example_images = glob('examples/*.png')

# loading the class map
class_map_path = "class_map.json"
with open("class_map.json","r") as f: classes = json.load(f)

# loading the encoding map (tensorflow dataset has encoded, i was also encoed, so 2 times encode issue)
with open("encoding_map.pickle","rb") as f: encodings = pickle.load(f)

# param declerations
class_size=len(classes.values())

# preparing the image for model
def process_image(image):
    # Convert into tensor
    image = tf.convert_to_tensor(image)

    # Cast the image to tf.float32
    image = tf.cast(image, tf.float32)
    
    # Resize the image to img_resize
    image = tf.image.resize(image, (30,30))
    
    # Normalize the image
    image = tf.keras.layers.Rescaling(1./255)(image)
    
    # Return the processed image and label
    return image

def decode_the_label(encoding_index):
  class_id = encodings[encoding_index] #encoding
  class_name = classes[class_id]
  return class_name

# Classifying the image
def predict(image):

  # Pre-procesing the data
  images = process_image(image)

  # Batching
  batched_images = tf.expand_dims(images, axis=0)
  
  prediction = model.predict(batched_images).flatten()
  confidences = {decode_the_label(i): np.round(float(prediction[i]), 3) for i in range(class_size)}
  return confidences

# creating the component
demo = gr.Interface(fn=predict, 
             inputs=gr.Image(shape=(30, 30)),
             outputs=gr.Label(num_top_classes=5),
             examples=example_images)
            
# Launching the demo
if __name__ == "__main__":
    demo.launch()
