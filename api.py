from fastapi import FastAPI, Request
from fastapi.responses import Response
import cv2
import numpy as np
from ultralytics import YOLO
import os
import gdown
import torch
import gc

# ==========================================
# 1. PUT PYTORCH ON A MEMORY DIET
# Stops Render from crashing on the Free Tier
# ==========================================
torch.set_num_threads(1) 

# ==========================================
# 2. PYTORCH 2.6 SECURITY BYPASS
# Tells the server to trust our YOLO model
# ==========================================
original_load = torch.load
def custom_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = custom_load

# 3. Download the heavy model from Google Drive
model_file = "best.pt"
if not os.path.exists(model_file):
    print("Downloading model from Google Drive...")
    gdown.download(id="1uV8IMuGDbmDabdjyeSy4SUKV9OS-ULbe", output=model_file, quiet=False)

# Start the app and load the model
app = FastAPI()
model = YOLO(model_file) 

@app.post("/anonymize")
async def anonymize_image(request: Request):
    # Catch the photo
    contents = await request.body()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return Response(status_code=400, content="Invalid image")

    # Find faces and plates
    results = model(img, conf=0.4, device='cpu')

    # Blur them
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            y1, y2 = max(0, y1), min(img.shape[0], y2)
            x1, x2 = max(0, x1), min(img.shape[1], x2)
            
            if x1 < x2 and y1 < y2:
                roi = img[y1:y2, x1:x2]
                blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
                img[y1:y2, x1:x2] = blurred_roi

    # Save blurred image
    _, encoded_img = cv2.imencode('.jpg', img)
    
    # 4. TAKE OUT THE TRASH (Forces Python to free up RAM)
    del img
    del results
    gc.collect()

    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")
