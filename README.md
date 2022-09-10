# NatuReddit
Small personal project aimed at developing a ML service which resembles a production environment system, from data collection, to data organization, model developement to finally deployement.

This focus of this project is not on the model (it's going to just be a text/image classification) but moreso on MLops.

## The project

The idea is to develop a service which is able to tell whether a reddit post is of fauna, flora or landscape.  
### Technologies
The MLops/DevOps tools used were:
- AirFlow
- MLFlow
- DVC
- Most of TFX
- TorchServe

The cloud deployment was done on AWS.

# Data collection
First of all we needed to obtain the data from reddit.  
The following subreddits have been chosen initially for scraping: wildlifephotography, naturepics, naturephotography, LandscapePhotography.  
The scraping was performed using the PushShift API (to gather posts in a given time interval) and PRAW was used to obtain the images from the posts.

The images needed to be resized to a reasonable size to not take up too much space.  
A dataset is defined at this point as a .csv file and a folder containing its images.

## Airflow Automation

An airflow pipeline was defined to obtain new submissions for the last day from said subreddits. Such a pipeline should be run every day.
Its steps are as follows:
- Obtain the latest dataset version from dvc
- Load data scraping settings, this steps also helps in identifying what was the last dataset version
- Obtain the posts from the latest day and their images
- Resize the images
- Merge the old dataset with the new one
- Update the dataset in dvc, also updating the latest dataset record entry
- Push updated .dvc files to github  

Airflow was used in a docker container.
## Using DVC & DagsHub
DVC is a data versioning system, in this project it was used to version both the .csv and the image folder.  
The DagsHub platform was used for remote storage, and it was amazing.

# TFX pipeline definition
The training pipeline uses all the classical steps defined by TFX except for the Evaluator step since there is no documentation on how to use a non-TF model to carry it out.  
This is a huge problem, sure you can run tests during the trainer but then the pusher wouldn't know what to do and you would need to do manual stuff.

As for the ExampleGen component, it's very inflexible as it requires the path you give it to only contain files like those in the dataset (.csv) in our case. For this reason the dataset was moved in another folder and the image folder was passed to the trainer later.

As for the trainer since we use a pytorch model we needed to define a custom Trainer.
In the custom trainer we create a pytorch Dataset which sources data from the tensorflow dataset provided by ExampleGen (such a step was needed anyway since the examples have to be connected to images).

The output of the trainer (i.e. a label mapping, the parameter dict and the model configuration) is then dumped into the "serving_model_dir", which the Pusher takes care to save in another easier to access directory.

Actually it looks like TFX treats dataset splits very badly, not a good look.  
It looks like a TFX pipeline can be run in airflow, in this project we just implemented a local DAG, but deploying it in airflow shouldn't be that bad
## MLFlow usage
MlFlow is an experiment (and model, but we didn't use it) tracking framework.  
What we did with it was logging a training run executed in the TFX pipeline, in practice we logged parameters (both about the model and the training routine) and training metrics.  

MlFlow data was sent to a remote storage hosted on DagsHub.
To do so we just needed to set some environment variables.
# Deployment with torchserve
Torchserve aids in serving a pytorch model and it also allows us to manage different model versions and types.  

Automation is actually missing between model training and creation of a deployable image. This is actually fine in this project since the evaluator is missing and manual intervention is needed anyway. 

To deploy with torchserve we first need to define a custom serve handler which has to define:
- How to load the model in the initialization step
- How to preprocess data (using HF preprocessing components, also stored in the model), both textual and image
- How to perform inference (pass preprocessed data correctly to the model)
- How to postprocess the model output (map the result back into a human-readable label)

We then need to package all the files needed for model creation and inference into a .mar file using torch-model-archiver.  
Those are:
- The file containing the model definition
- The  parameter dict
- The file that defines the custom serve handler
- Extra files like model config and label mappings

Finally we can run the server with the following command:
```
torchserve 
    --start
    --model-store torchserve/models
    --ts-config torchserve/config.properties
    --models naturedditTI=naturedditTI.mar
```
We can send a prediction POST request in the following way:
```
curl https://127.0.0.1/predictions/naturedditTI -F "text=<text>" -F "image=@<img path>
```
Careful, HTTPS is needed so force curl to use it.

Torchserve was then packaged into a docker container. There is a supposed easier method to create it but then you can't inject dependencies.  
Anyway we created a Dockerfile extending the "pytorch/torchserve:0.6.0-cpu" image, installing the required libraries for our model with an ad-hoc requirements file and finally we copied the model to serve.
# Deployment on the cloud

It was decided to use AWS to deploy the service on the cloud. Since we created a docker container we can just use it for deployement.  

The first step is to push this docker container to ECR (Elastic container registry).  

Then we can create a ECS (Elastic container service). A new cluster based on a fresh EC2 instance (t4.small) was created.  
We then needed to define a new task to run on the ECS, this task was defined through the docker container and it exposed the necessary ports.  
Finally the task was ran.  

It has been possible to access the task from outside by accessing the DNS of the underlying instance.  
To make sure that requests were accepted it was necessary to set security groups inbound rules for the needed ports.  
