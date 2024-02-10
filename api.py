from flask import Flask, request
from flask import jsonify
import base64
import uuid
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
model = tf.keras.models.load_model('cnn_model.h5')

@app.route('/api/', methods=['GET', 'POST'])
def index():
    print("Request received")
    data = request.json

    file_name = save_image_content(data['imageBase64'])
    result = process_image_content(file_name)

    return jsonify(result)


def save_image_content(encoded_data):
  decoded_data=base64.b64decode((encoded_data))
  file_name = uuid.uuid4().hex
  img_file = open('uploads/'+file_name+'.jpg', 'wb')
  img_file.write(decoded_data)
  img_file.close()
  return 'uploads/'+file_name+'.jpg'

def predict_class(image, model):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [512, 512])
    image = np.expand_dims(image, axis = 0)
    prediction = model.predict(image)
    return prediction
    
def process_image_content(file_name):
  test_image = Image.open(file_name)
  
  pred = predict_class(np.asarray(test_image), model)
  class_names = ['black_sigatoka', 'fusarium_wilt', 'healthy', 'not_banana']
  
  data = (pred.tolist()[0])
  index = np.argmax(pred)
  result = {}
  result['class_name'] = class_names[index]
  result['confidence_score'] = data[index] * 100
  return(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)