from google.cloud import vision
from google.cloud.vision import types
from pathlib import Path
from google.protobuf.json_format import MessageToJson
import io
import os
import shutil
from PIL import Image
import json
import time

def detect_face(input_file):
  from google.cloud import vision
  client = vision.ImageAnnotatorClient()

  content = input_file.read()
  image = types.Image(content=content)

  return client.face_detection(image=image)

def crop_face(image, faces, output_filename):
  im = Image.open(image)
  for face in faces:
    position = [(vertex.get("x") if vertex.get("x") is not None else 0, vertex.get("y") if vertex.get("y") is not None else 0)
      for vertex in face.get("boundingPoly").get("vertices")]
    im_crop_outside = im.crop((position[0][0], position[0][1], position[2][0], position[2][1]))

  if im_crop_outside.mode != "RGB":
    im_crop_outside = im_crop_outside.convert("RGB")

  im_crop_outside.save(output_filename, quality=95)

def main():
  input_path = "C:\\temp\\inputImages"
  output_path = "C:\\temp\\outputImages"

  p = Path(input_path)
  input_files = list(sorted(p.glob('*.jpg'), key=os.path.getmtime))

  for input_file in input_files:
    with open(input_file, 'rb') as image:
      json_file = Path(output_path, input_file.stem+".json")
      if not json_file.exists():
        with open(json_file, 'w') as f:
          response = detect_face(image)
          json_file.write_text(MessageToJson(response))

  json_files = list(sorted(Path(output_path).glob('*.json'), key=os.path.getmtime))
  for json_file in enumerate(json_files):
    with open(json_file, 'rb') as data_file:
      json_data = json.loads(data_file.read().decode('utf-8')).get("faceAnnotations")
      if not json_data:
        continue
      image = Path(input_path, json_file.stem+".jpg")
      output_image = Path(output_path, json_file.stem+".jpg")
      crop_face(image, json_data, output_image)

if __name__ == "__main__":
  main()