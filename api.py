import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MALLOC_TRIM_THRESHOLD_"] = "10000"

from fastapi import FastAPI, Request
from fastapi.responses import Response
import cv2
import numpy as np
import onnxruntime as ort
import gc
import traceback

app = FastAPI()

print("Loading raw ONNX engine (Zero PyTorch Bloat)...")
session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 1
session_options.inter_op_num_threads = 1
session = ort.InferenceSession("best.onnx", sess_options=session_options)
input_name = session.get_inputs()[0].name

print("Loading High-Res Face Detector Safety Net...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.get("/")
def health_check():
    return {"status": "The Python server is fully awake and listening on the correct port!"}

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

        img_h, img_w = img.shape[:2]
        
        # 1. Pure Math Pre-processing (ONNX Model is locked to 320x320)
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (320, 320), swapRB=True, crop=False)
        
        # 2. Raw ONNX Inference
        outputs = session.run(None, {input_name: blob})
        preds = outputs[0][0].T 
        
        # 3. Pure Math Post-Processing
        x_factor = img_w / 320.0
        y_factor = img_h / 320.0
        
        boxes =[]
        confidences =[]
        
        # --- LAYER 1: Scan ONNX Results ---
        for row in preds:
            scores = row[4:]
            max_score = np.max(scores)
            if max_score > 0.20: 
                x_c, y_c, w, h = row[0:4]
                left = int((x_c - w / 2) * x_factor)
                top = int((y_c - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                boxes.append([left, top, width, height])
                confidences.append(float(max_score))

        # --- LAYER 2: HIGH-RES SAFETY NET ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # ANTI-CRASH FIX: Downscale the grayscale image for the face detector to prevent RAM spike
        max_dim = 640.0
        scale_down = 1.0
        if max(img_h, img_w) > max_dim:
            scale_down = max_dim / max(img_h, img_w)
            new_w = int(img_w * scale_down)
            new_h = int(img_h * scale_down)
            gray_small = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            gray_small = gray

        # scaleFactor=1.2 uses exponentially less RAM than 1.1
        faces = face_cascade.detectMultiScale(gray_small, scaleFactor=1.2, minNeighbors=4, minSize=(20, 20))
        
        for (x, y, w, h) in faces:
            # Map the coordinates back to the original large image
            real_x = int(x / scale_down)
            real_y = int(y / scale_down)
            real_w = int(w / scale_down)
            real_h = int(h / scale_down)
            boxes.append([real_x, real_y, real_w, real_h])
            confidences.append(0.95)
            
        del gray
        if 'gray_small' in locals():
            del gray_small
                
        # 4. Filter overlapping boxes and Apply Blur
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.15, 0.4)
        
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(img_w, x + w)
                y2 = min(img_h, y + h)
                
                if x1 < x2 and y1 < y2:
                    roi = img[y1:y2, x1:x2]
                    h_roi, w_roi = roi.shape[:2]
                    if h_roi > 0 and w_roi > 0:
                        small = cv2.resize(roi, (8, 8), interpolation=cv2.INTER_LINEAR)
                        pixelated = cv2.resize(small, (w_roi, h_roi), interpolation=cv2.INTER_NEAREST)
                        blurred_roi = cv2.GaussianBlur(pixelated, (15, 15), 10)
                        img[y1:y2, x1:x2] = blurred_roi

        # 5. Clean up memory
        del preds, outputs, blob
        gc.collect()

        _, encoded_img = cv2.imencode('.jpg', img)
        del img 
        gc.collect()

        return Response(content=encoded_img.tobytes(), media_type="image/jpeg")
        
    except Exception as e:
        error_msg = traceback.format_exc()
        print(error_msg)
        return Response(status_code=500, content=f"Python Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
