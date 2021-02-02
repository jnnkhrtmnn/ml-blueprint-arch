# ml-blueprint-arch
 blueprint folder structure and Flask setup for machine learning projects using a prediction API endpoint.

## Conda envs
Firt, create an environment with
`conda create -n blueprint python=3.8`,\
activate the environment using
`conda activate blueprint`,\
then `cd` to the root folder of your project and install all requirements with 
`conda install -r requirements.txt`

## Data & Training
In the same cd execute `python src/generate_data.py`\
and `python src/train.py`


## Run Flask App
To run the app, simply execute `python app.py`


## API test
Test the API by sending the stored POST request using POSTMAN
