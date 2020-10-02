import os
import json
import os

import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request

from constants import req_labels

app = Flask(__name__)

model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

model.eval()

def transform_image(file):
  image = Image.open(file)
  transformed_img = transforms.ToTensor()(image).unsqueeze_(0)
  return transformed_img


def get_prediction(input_tensor):
  output = model(input_tensor)[0]
  boxes = output['boxes'].detach().numpy()
  scores = output['scores'].detach().numpy()
  labels = output['labels'].numpy()
  predicted_class = [{"label": req_labels[label].lower(), "score": np.around(scores[index], decimals=2)*100} for index, label in enumerate(labels)]
  predicted_class = list(filter(lambda cls: (cls['label'] != '') and (cls['score'] > 0.4), predicted_class))
  print(predicted_class)
  return predicted_class


@app.route('/predict', methods=['POST'])
def predict():
  if request.method == 'POST':
    print("Request recieved...")
    file = request.files['file']
    if file is not None:
      input_tensor = transform_image(file)
      print("Converted Image successfully to tensor, getting predictions...")
      prediction = get_prediction(input_tensor)
      return jsonify(prediction)


if __name__ == '__main__':
    app.run()
