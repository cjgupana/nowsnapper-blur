import os
# ==========================================
# 1. LINUX MEMORY HACKS (MUST BE AT THE VERY TOP)
# Prevents the server from hoarding RAM
# ==========================================
os.environ["MALLOC_ARENA_MAX"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"

from fastapi import FastAPI, Request
from fastapi.responses import Response
import cv2
import numpy as np
import gdown
import traceback
import gc
import torch
from ultralytics import YOLO

# ==========================================
# 2. KILL PYTORCH MEMORY TRACKING
# ==========================================
torch.set_grad_enabled(False)
torch.set_num_threads(1)

original_load = torch.load
def custom_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = custom_load

model_file = "best.pt"
if not os.path.exists(model_file):
    print("Downloading model...")
    gdown.download(id="1uV8IMuGDbmDabdjyeSy4SUKV9OS-ULbe", output=model_file, quiet=False)

app = FastAPI()
model = YOLO(model_file, task='detect')

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

        # ==========================================
        # 3. WITH TORCH.NO_GRAD()
        # This is the magic lock that stops PyTorch 
        # from allocating hundreds of MBs of RAM!
        # ==========================================
        with torch.no_grad():
            results = model(img, conf=0.4, device='cpu', imgsz=320)

        # Apply Blur
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
