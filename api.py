from fastapi import FastAPI, Request
from fastapi.responses import Response
import cv2
import numpy as np
from ultralytics import YOLO
import gc
import traceback
import os
import gdown

# 1. Download the ONNX model from Google Drive
model_file = "best.onnx"
if not os.path.exists(model_file):
    print("Downloading ONNX model from Google Drive...")
    # ==========================================
    # PASTE YOUR NEW GOOGLE DRIVE ID HERE:
    # ==========================================
    gdown.download(id="1f2zrlb9r1vqO5aX8rYCjQxF78-MT2R1k", output=model_file, quiet=False)

# 2. Load the ultra-lightweight ONNX model
model = YOLO(model_file, task='detect') 

app = FastAPI()

@app.post("/anonymize")
async def anonymize_image(request: Request):
    try:
        gc.collect() 
        
        contents = await request.body()
        nparr = np.frombuffer(contents, np.uint8)
        del contents 
        
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        del nparr 
        
        if img is None:
            return Response(status_code=400, content="Invalid image format")

        # Run ONNX inference (uses almost zero RAM!)
        results = model(img, conf=0.4, imgsz=320)

        for result in results:
            # Safely extract boxes whether it's a tensor or numpy array
            boxes = result.boxes.xyxy
            if hasattr(boxes, 'cpu'):
                boxes = boxes.cpu().numpy()
            else:
                boxes = np.array(boxes)
                
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                y1, y2 = max(0, y1), min(img.shape[0], y2)
                x1, x2 = max(0, x1), min(img.shape[1], x2)
                
                if x1 < x2 and y1 < y2:
                    roi = img[y1:y2, x1:x2]
                    blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
                    img[y1:y2, x1:x2] = blurred_roi

        del results
        gc.collect()

        _, encoded_img = cv2.imencode('.jpg', img)
        del img 
        gc.collect()

        return Response(content=encoded_img.tobytes(), media_type="image/jpeg")
        
    except Exception as e:
        error_msg = traceback.format_exc()
        print(error_msg)
        return Response(status_code=500, content=f"Python Error: {str(e)}")
