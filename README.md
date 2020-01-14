# Environment Sound Classification

## Background

This project was created in order to reproduce the results obtained from the sensors_paper.pdf (cite properly), the paper itself conatains a few inconsistencies but was used to successfully recreate the convolutional neural network (CNN). 

The network classifies 10 different recorded sounds:
* air_conditioner
* car_horn
* children_playing
* dog_bark 
* drilling
* engine_idling
* gun_shot
* jackhammer 
* siren 
* street_music

## Running the Code

To run the code, the model used to train the classifier needs to be specified these are:
* LMC
* MC
* MLMC
* TSCNN

An example of training and validating the CNN with the LMC model is given below:

`$ python neuralnet.py LMC`
