I decide to Use Fully ONNX runtime, Postgress and Docker 

FIrst git clone This Repo
In Shell path on your directory Write This Command 

Docker Compose  

docker-compose up --build


Open http://127.0.0.1:8000/docs#/ and there is FastAPI-Swagger UI to Test the API

build Docker image (Not Recomended, Because Library Deprecancy within Another System, because my sytem use Python 3.10, better use Compose. So we can get the db still virtually within docker image. So will never run to localy system but use compose to still run inside docker)


docker build -t face-recognition-app .


docker run -p 8000:8000 face-recognition-app


After Finished
Open This Local Link 
http://127.0.0.1:8000/docs#/
and there is FastAPI-Swagger UI to Test the API

The Test Above Can upload an Image. As a test you can use what ever image. I include some image in folder Face_embed
Thank You
