from fastapi import FastAPI, Request
from fastapi.responses import Response
import cv2
import numpy as np
from ultralytics import YOLO
import os
import gdown
import torch
import gc
import traceback

torch.set_num_threads(1) 

original_load = torch.load
def custom_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = custom_load

model_file = "best.pt"
if not os.path.exists(model_file):
    print("Downloading model from Google Drive...")
    gdown.download(id="1uV8IMuGDbmDabdjyeSy4SUKV9OS-ULbe", output=model_file, quiet=False)

app = FastAPI()
model = YOLO(model_file) 

@app.post("/anonymize")
async def anonymize_image(request: Request):
    try:
        # Wipe memory clean before we even start
        gc.collect() 
        
        # 1. Read bytes and immediately delete the original from RAM
        contents = await request.body()
        nparr = np.frombuffer(contents, np.uint8)
        del contents 
        
        # 2. Decode image and immediately delete the numpy array
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        del nparr 
        
        if img is None:
            return Response(status_code=400, content="Invalid image format")

        # 3. Squint AI (imgsz=320) to save RAM
        results = model(img, conf=0.4, device='cpu', imgsz=320)

        # 4. Apply Blur
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

        # 5. Delete YOLO results from RAM before we encode
        del results
        gc.collect()

        # 6. Encode image to JPG
        _, encoded_img = cv2.imencode('.jpg', img)
        
        # 7. Delete the massive image canvas from RAM before sending!
        del img 
        gc.collect()

        return Response(content=encoded_img.tobytes(), media_type="image/jpeg")
        
    except Exception as e:
        error_msg = traceback.format_exc()
        print(error_msg)
        return Response(status_code=500, content=f"Python Error: {str(e)}")
