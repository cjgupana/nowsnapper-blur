from fastapi import FastAPI, Request
from fastapi.responses import Response
import cv2
import numpy as np
from ultralytics import YOLO
import os
import gdown
import torch

# ==========================================
# PYTORCH 2.6 SECURITY BYPASS
# Tells the server to trust our YOLO model
# ==========================================
original_load = torch.load
def custom_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = custom_load
# ==========================================

# 1. Download the heavy model from Google Drive automatically!
model_file = "best.pt"
if not os.path.exists(model_file):
    print("Downloading model from Google Drive...")
    gdown.download(id="1uV8IMuGDbmDabdjyeSy4SUKV9OS-ULbe", output=model_file, quiet=False)

# 2. Start the app and load the model
app = FastAPI()
model = YOLO(model_file) 

@app.post("/anonymize")
async def anonymize_image(request: Request):
    # Catch the photo from WordPress
    contents = await request.body()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return Response(status_code=400, content="Invalid image")

    # Find the faces and license plates
    results = model(img, conf=0.4, device='cpu')

    # Put a heavy blur over them
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            
            # Make sure we don't blur outside the photo edges
            y1, y2 = max(0, y1), min(img.shape[0], y2)
            x1, x2 = max(0, x1), min(img.shape[1], x2)
            
            if x1 < x2 and y1 < y2:
                roi = img[y1:y2, x1:x2]
                blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
                img[y1:y2, x1:x2] = blurred_roi

    # Throw the safe photo back to WordPress
    _, encoded_img = cv2.imencode('.jpg', img)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")
