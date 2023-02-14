# 自己写的，瞎搞的

from ultralytics import YOLO
from PIL import Image
import cv2

# # Train
# model = YOLO("yolov8n.pt") # pass any model type
# model.train(epochs=5)

# # Val
# model = YOLO("yolov8n.yaml")
# model.train(data="coco128.yaml", epochs=5)
# model.val()  # It'll automatically evaluate the data you trained.
#
#Predict
model = YOLO("yolov8n.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source="0")
# results = model.predict(source="folder", show=True) # Display preds. Accepts all YOLO predict arguments

# from PIL
im1 = Image.open("bus.jpg")
results = model.predict(source=0, stream=False, save=True)  # save plotted images

# from ndarray
im2 = cv2.imread("zidane.jpg")
results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels
for result in results:
    # detection
    result.boxes.xyxy  # box with xyxy format, (N, 4)
    result.boxes.xywh  # box with xywh format, (N, 4)
    result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
    result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
    result.boxes.conf  # confidence score, (N, 1)
    result.boxes.cls  # cls, (N, 1)
print(result.boxes.xyxy, result.boxes.xyxyn)

# from list of PIL/ndarray
results = model.predict(source=[im1, im2])

# #Export
#
# model = YOLO("model.pt")
# model.fuse()
# model.info(verbose=True)  # Print model information
# model.export(format=)  # TODO:

