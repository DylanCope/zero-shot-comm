# Learning to Communicate with Strangers via Channel Randomisation Methods

This is the source code for our paper "Learning to Communicate with Strangers via Channel Randomisation Methods" (Cope and Shoots, 2020) first informally presented at the 4th NeurIPS Workshop on Emergent Communication.

## Installation

There are two ways of installing and running this codebase. 

* One way is to use the `environment.yml` to recreate the conda environment.

* Another way is to create a Docker container using the provided Dockerfile and scripts. The image inherits from `tensorflow/tensorflow:latest-gpu-jupyter`. If you do not have gpu access then you will need to adapt the Dockerfile and change the run script (remove the `--runtime nvidia` parameter).

