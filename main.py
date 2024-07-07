from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import numpy as np
import cv2
import onnxruntime as ort
from sklearn.metrics.pairwise import cosine_similarity
import io
import json
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the yunet ONNX model for face detection
weights = "face_detection_yunet_2023mar.onnx"
face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))

# Initialize ONNX runtime for face embedding model
embedding_model_path = 'faceNet.onnx'
embedding_session = ort.InferenceSession(embedding_model_path)

# Temporary storage for face embeddings
known_faces = []

# Function to preprocess and extract face embeddings
def extract_face_embeddings(face_image):
    input_size = (640, 640)
    resized_image = cv2.resize(face_image, input_size, interpolation=cv2.INTER_LINEAR)
    face_detector.setInputSize(input_size)
    _, detections = face_detector.detect(resized_image)

    if detections is None:
        logging.debug("No detections found.")
        return None

    face_embeddings = []
    for detection in detections:
        confidence = detection[-1]
        if confidence > 0.6:
            x1, y1, width, height = detection[:4].astype(int)
            x2, y2 = x1 + width, y1 + height
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(resized_image.shape[1], x2), min(resized_image.shape[0], y2)
            cropped_face = resized_image[y1:y2, x1:x2]
            target_size = (160, 160)
            resized_face = cv2.resize(cropped_face, target_size, interpolation=cv2.INTER_CUBIC)
            face_pixels = resized_face.astype(np.float32) / 255.0

            if face_pixels.shape[-1] != 3:
                logging.debug("Face image does not have 3 channels. Sesuai format facenet")
                continue

            face_pixels = np.expand_dims(face_pixels, axis=0)
            input_name = embedding_session.get_inputs()[0].name
            output_name = embedding_session.get_outputs()[0].name
            embeddings = embedding_session.run([output_name], {input_name: face_pixels.astype(np.float32)})[0]
            face_embeddings.append(embeddings.flatten())

    return face_embeddings

class Face(BaseModel):
    id: int
    embedding: list
    image_path: str

@app.get("/api/face", response_model=list[Face])
def get_faces():
    return known_faces

@app.post("/api/face/register")
async def register_face(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        face_image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        embeddings = extract_face_embeddings(face_image)
        if not embeddings:
            raise HTTPException(status_code=400, detail="No face detected")

        global known_faces
        face_id = len(known_faces) + 1
        for emb in embeddings:
            known_faces.append(Face(id=face_id, embedding=emb.tolist(), image_path=file.filename))
            face_id += 1

        return {"message": "Face registered successfully"}
    except Exception as e:
        logging.exception("Error registering face")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/face/recognize")
async def recognize_face(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        face_image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        embeddings = extract_face_embeddings(face_image)
        if not embeddings:
            raise HTTPException(status_code=400, detail="No face detected")

        if not known_faces:
            raise HTTPException(status_code=404, detail="No faces registered")

        similarities = []
        for emb in embeddings:
            for face in known_faces:
                known_embedding = np.array(face.embedding)
                similarity = cosine_similarity([emb], [known_embedding])[0][0]
                similarities.append((face.id, similarity))

        best_match = max(similarities, key=lambda x: x[1])
        if best_match[1] > 0.65:  #Treshold For Similiarity Cosine from sklearn.metrics.pairwise import cosine_similarity, More Higher means more need to spesific but higher accuracy
            matched_face = next((face for face in known_faces if face.id == best_match[0]), None)
            if matched_face:
                return matched_face

        return {"message": "No matching face found"}
    except Exception as e:
        logging.exception("Error recognizing face")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/face/{id}")
def delete_face(id: int):
    try:
        global known_faces
        known_faces = [face for face in known_faces if face.id != id]
        return {"message": "Face deleted successfully"}
    except Exception as e:
        logging.exception("Error deleting face")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)