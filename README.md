# Environment Sound Classification

## Background

This project was created in order to reproduce the results obtained from the sensors_paper.pdf [1]. the paper itself conatains a few inconsistencies but was used to successfully recreate the convolutional neural network (CNN). 

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

```
$ python neuralnet.py LMC
```
## References
<a id="1">[1]</a> 
Y. Su, K. Zhang, J. Wang, and K. Madani, “Environment soundclassification  using  a  two-stream  cnn  based  on  decision-levelfusion,”Sensors, vol. 19, p. 1733, 04 2019.
