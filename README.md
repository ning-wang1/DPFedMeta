# DPFedMeta
Code for paper 'Squeezing More Utility via Adaptive Clipping on Deferentially Private Gradients in Federated Meta-Learning' ACSAC 2022

## The running platform
We ran all our experiments on a server equipped with a 3.3 GHz Intel Core i9-9820X CPU, three GeForce RTX 2080 Ti GPUs, and Ubuntu 18.04.3 LTS. we are using  Pytorch 1.4.0 python, python 3.6.10.


## prepare the environment
1. Implement annoconda following the instruction in https://www.anaconda.com/. Anaconda will help to manage the learning environement.
2.  Create your env using the yml file by the command line: conda env create -n envname --file environment.yml. All the required libraries (including pytorch and python) will be implemented. The environment is named as 'myenv' as specified in the yml file.
3. activate your environment. Use: conda activate envname

## Run the experiment 
1. go to the folder experiment_scripts: cd experiment_scripts
2. run the training 
Markup: * omniglot dataset use omniglot_5_8_0.1_64_5_0_few_shot.sh
        * imagenet dataset use bash mini-imagenet_5_2_0.01_48_5_0_few_shot.sh 
        * cifar-100 dataset use cifar-fs_5_8_0.01_48_5_0_few_shot
