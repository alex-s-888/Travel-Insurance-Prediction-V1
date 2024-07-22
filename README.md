# Travel Insurance Prediction V1

The goal of the project is to build and deploy Machine Learning model that predicts if customer will buy travel insurance policy based on customer's profile. 

Imagine following use case: company is planning promo campaign where it will send some gifts to encourage customers buy policies. 
The budget of campaign is limited so gifts should be sent only to customers that are likely to do the purchase.

## Project structure

### dataset
Folder contains [TravelInsurancePrediction.csv](dataset/TravelInsurancePrediction.csv) spreadsheet with customers' data.

### model
Contains the code [train.py](model/train.py) that builds the model.

### deployment
Folder contains files necessary to deploy model as Docker image.

### test
Instructions how to use/test predictions using dockerized model. 


## Prerequisites

This is python (ver 3.11) project. The following packages should be present:
- pandas
- numpy
- scikit-learn
- mlflow
- joblib

Docker should be installed as it is used to run mlflow server and deploy/run resulting model.  


## Walkthrough

First, mlflow server should be launched. I was using `atcommons/mlflow-server` docker image for this purpose. The command to run  
(note: environment variable `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` is used to avoid potential error in protobuf library):  
`docker run -p 5000:5000 --env PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python atcommons/mlflow-server`  
Mlflow UI should be available at `http://localhost:5000`

Next step is to execute `model/train.py` code. It will do the following:
- read dataset and apply necessary data transformations, clean-up, etc...
- split data to Train and Test sets
- do several runs with different hyperparameters using `GradientBoostingClassifier`
- track experiments using mlflow server
- build model based on best hyperparameters values
- serialize model using `joblib` format, resulting file `my_model.joblib` will be located in `deployment` folder
- (mlflow) register model/version and mark state as `Staging`

After the model is built, it may be deployed as Docker image. See instructions [deployment/README.md](deployment/README.md) how to use `Dockefile` and other stuff.

The dockerized model may be used to execute batch prediction jobs, see example [test/README.md](test/README.md).

&nbsp;  
Screenshot examples of mlflow UI:  

![mlflow_001.png](screenshots%2Fmlflow_001.png)   

&nbsp;    

![mlflow_002.png](screenshots%2Fmlflow_002.png)
