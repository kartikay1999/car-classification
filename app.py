from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

model=load_model('cars1.h5')
app = Flask(__name__)
  
from werkzeug import secure_filename

graph = tf.get_default_graph()

@app.route('/upload')
def upload_file():
    return render_template('index.html')
@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        print(f.filename)
        test_image=image.load_img(f.filename,target_size=(64,64))
        test_image=image.img_to_array(test_image)
        test_image=np.expand_dims(test_image,axis=0)
        output_dict={0: 'convertible',1: 'coupe',2: 'electric', 3: 'jeep',4: 'luxury',5: 'midsize',6: 'military',7: 'royal',8: 'sports',9: 'suv'}
        with graph.as_default():
            y = model.predict_classes(test_image)
            out=y[0]
    return render_template("prediction.html",result = {'prediction':output_dict[out]})



if __name__ == "__main__":
    app.run(debug=False)
