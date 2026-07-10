
# Start local MLflow tracking server
mlflow server --host 127.0.0.1 --port 8080

# deploying a model locally as an Inference Server
mlflow models serve -m models:/RidgeClassifier/latest -p 1234 --enable-mlserver
mlflow models serve -m models:/m-6537703297e8417288fe7fff724f04bf -p 1234 --enable-mlserver

# build a docker image for mlflow model
mlflow models build-docker -m models:/m-6537703297e8417288fe7fff724f04bf -n albanacelepija/hiring-sklearn-mlflow --enable-mlserver
# docker push albanacelepija/hiring-sklearn-mlflow

