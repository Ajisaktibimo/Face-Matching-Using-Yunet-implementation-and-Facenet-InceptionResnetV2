from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from sqlalchemy.orm import Session
import numpy as np
import cv2
import onnxruntime as ort
from sklearn.metrics.pairwise import cosine_similarity
import logging

from db_models import Base, FaceModel, SessionLocal, engine
import schemas

app = FastAPI()
logging.basicConfig(level=logging.DEBUG)

weights = "face_detection_yunet_2023mar.onnx"
face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))

embedding_model_path = 'faceNet.onnx'
embedding_session = ort.InferenceSession(embedding_model_path)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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
        if confidence > 0.5:
            x1, y1, width, height = detection[:4].astype(int)
            x2, y2 = x1 + width, y1 + height
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(resized_image.shape[1], x2), min(resized_image.shape[0], y2)
            cropped_face = resized_image[y1:y2, x1:x2]
            target_size = (160, 160)
            resized_face = cv2.resize(cropped_face, target_size, interpolation=cv2.INTER_CUBIC)
            face_pixels = resized_face.astype(np.float32) / 255.0

            if face_pixels.shape[-1] != 3:
                logging.debug("Face image does not have 3 channels. Skipping...")
                continue

            face_pixels = np.expand_dims(face_pixels, axis=0)
            input_name = embedding_session.get_inputs()[0].name
            output_name = embedding_session.get_outputs()[0].name
            embeddings = embedding_session.run([output_name], {input_name: face_pixels.astype(np.float32)})[0]
            face_embeddings.append(embeddings.flatten())

    return face_embeddings

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)

@app.get("/api/face", response_model=list[schemas.FaceModel])
def get_faces(db: Session = Depends(get_db)):
    faces = db.query(FaceModel).all()
    return faces
@app.post("/api/face/register")
async def register_face(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        image_data = await file.read()
        face_image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        embeddings = extract_face_embeddings(face_image)
        if not embeddings:
            raise HTTPException(status_code=400, detail="No face detected")

        for emb in embeddings:
            face_entry = FaceModel(embedding=emb.tolist(), image_path=file.filename)
            db.add(face_entry)
        db.commit()

        return {"message": "Face registered successfully"}
    except Exception as e:
        logging.exception("Error registering face")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/face/recognize")
async def recognize_face(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        image_data = await file.read()
        face_image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        embeddings = extract_face_embeddings(face_image)
        if not embeddings:
            raise HTTPException(status_code=400, detail="No face detected")

        known_faces = db.query(FaceModel).all()
        if not known_faces:
            raise HTTPException(status_code=404, detail="No faces registered")

        similarities = []
        for emb in embeddings:
            for face in known_faces:
                known_embedding = np.array(face.embedding)
                similarity = cosine_similarity([emb], [known_embedding])[0][0]
                similarities.append((face.id, similarity))

        best_match = max(similarities, key=lambda x: x[1])
        if best_match[1] > 0.55:  # Assuming a threshold of 0.55 for a match Based On how far the Vector. Cosine Similiarity From Facenet based Network
            matched_face = db.query(FaceModel).filter(FaceModel.id == best_match[0]).first()
            if matched_face:
                return {"id": matched_face.id, "embedding": matched_face.embedding, "image_path": matched_face.image_path}

        return {"message": "No matching face found"}
    except Exception as e:
        logging.exception("Error recognizing face")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/face/{id}")
def delete_face(id: int, db: Session = Depends(get_db)):
    try:
        face_to_delete = db.query(FaceModel).filter(FaceModel.id == id).first()
        if face_to_delete:
            db.delete(face_to_delete)
            db.commit()
            return {"message": "Face deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Face not found")
    except Exception as e:
        logging.exception("Error deleting face")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
