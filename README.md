FIrst git clone This Repo
In Shell path on your directory Write This Command to 

Docker Compose  

docker-compose up --build


Open http://127.0.0.1:8000/docs#/ and there is FastAPI-Swagger UI to Test the API

build Docker image (Not Recomended, Because Library Deprecancy within my system, better use Compose. So we can get the db still virtually)


docker build -t face-recognition-app .


docker run -p 8000:8000 face-recognition-app


After Finished
Open This Local Link 
http://127.0.0.1:8000/docs#/
and there is FastAPI-Swagger UI to Test the API

The Test Above Can upload an Image
Thank You
