from pydantic import BaseModel
from typing import List

class FaceModelBase(BaseModel):
    embedding: List[float]
    image_path: str

class FaceModelCreate(FaceModelBase):
    pass

class FaceModel(FaceModelBase):
    id: int

    class Config:
        orm_mode: True
