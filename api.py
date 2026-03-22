import os
# ==========================================
# 1. KILL TELEMETRY & THREAD HOARDING
# Stops the app from hanging during boot!
# ==========================================
os.environ["YOLO_VERBOSE"] = "False" 
os.environ["ULTRALYTICS_ENV"] = "production"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from fastapi import FastAPI, Request
from fastapi.responses import Response
import cv2
import numpy as np
from ultralytics import YOLO
import gc
import traceback

app = FastAPI()

# We will "Lazy Load" the model so the server boots instantly
model = None

# ==========================================
# NEW: INSTANT HEALTH CHECK ROUTE
# ==========================================
@app.get("/")
def health_check():
    return {"status": "The Python server is fully awake and responding!"}

@app.post("/anonymize")
async def anonymize_image(request: Request):
    global model
    try:
        # ==========================================
        # 2. LAZY LOAD & CORRUPTION CHECK
        # Only loads the AI when a photo arrives!
        # ==========================================
        if model is None:
            if not os.path.exists("best.onnx"):
                return Response(status_code=500, content="Error: best.onnx is missing from the server!")
            
            # Check if Google Drive gave us a fake HTML page instead of the 30MB model
            if os.path.getsize("best.onnx") < 100000: 
                return Response(status_code=500, content="Error: best.onnx is corrupted! (File is too small). Google Drive likely blocked the download.")
            
            print("Loading ONNX model into memory...")
            model = YOLO("best.onnx", task='detect')
            print("Model loaded successfully!")

        gc.collect() 
        
        contents = await request.body()
        nparr = np.frombuffer(contents, np.uint8)
        del contents 
        
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        del nparr 
        
        if img is None:
            return Response(status_code=400, content="Invalid image format")

        # Run ONNX inference
        results = model(img, conf=0.4, imgsz=320)

        for result in results:
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
