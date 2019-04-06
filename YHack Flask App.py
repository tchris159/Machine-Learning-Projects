from flask import Flask, request
import json
import os
import random
import urllib.request
from scripts.label_image import *
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/run-tensor-flow", methods=["POST"]) #runs tensor flow at this address
def runTensorFlow():
    image = request.json['image'].encode("ascii", "ignore")
    return jsonify({"image": image})
    image_path = request.data       #requests the image file from front end
    if not image_path:
        return 'not valid'        #error message
    image_name = str(random.getrandbits(128)) + '.png'
    urllib.request.urlretrieve(image_path, image_name)
    tensor_flow_results = run_label_image(os.getcwd() + '/' + image_name, os.getcwd() + '/tf_files/optimized_graph.pb')    #runs scropts from terminal to perform machine learning model
    return json.dumps(tensor_flow_results)  #returns the confidence index of our machine learning model

app.run(debug=True)