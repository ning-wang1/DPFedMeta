# DPFedMeta
Code for paper 'Squeezing More Utility via Adaptive Clipping on Deferentially Private Gradients in Federated Meta-Learning' ACSAC 2022

## The running platform
We ran all our experiments on a server equipped with a 3.3 GHz Intel Core i9-9820X CPU, three GeForce RTX 2080 Ti GPUs, and Ubuntu 18.04.3 LTS. we are using  Pytorch 1.4.0 python, python 3.6.10.


## Prepare the environment
1. Implement annoconda following the instruction in https://www.anaconda.com/. Anaconda will help to manage the learning environement.
2.  Create your env using the yml file. All the required libraries (including pytorch and python) will be implemented. The environment is named as 'myenv' as specified in the yml file.
-      conda env create --file environment.yml
3. activate your environment. 
-      conda activate envname

## Prepare the data
We use three datasets, including Omniglot [1], CIFAR-FS [2], and Mini-ImageNet [3]. For convenience, the link of the data are listed:

[omniglot dataset](https://www.omniglot.com/)

[CIFAR-FS dataset](https://drive.google.com/file/d/1pTsCCMDj45kzFYgrnO67BWVbKs48Q3NI/view)

[mini-ImageNet dataset](https://drive.google.com/file/d/1R6dA6QGEW-lmiNkitCwK4IkAbl4uT3y3/view)

[1] Brenden M Lake et al. 2015. Human-level concept learning through probabilistic program induction. Science 350, 6266 (2015), 1332–1338

[2] Luca Bertinetto, Joao F Henriques, Philip HS Torr, and Andrea Vedaldi. 2019. Meta-learning with differentiable closed-form solvers. In International Conference on Learning Representations (ICLR 19)

[3] Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Daan Wierstra, et al . 2016. Matching networks for one shot learning. In Advances in neural information processing systems (NeurIPS 16). 3630–3638

## Run the experiment 
1. go to the folder experiment_scripts: cd experiment_scripts
2. run the training 
-      omniglot dataset: bash omniglot_5_8_0.1_64_5_0_few_shot.sh
-      imagenet dataset: bash mini-imagenet_5_2_0.01_48_5_0_few_shot.sh 
-      cifar-100 dataset: bash cifar-fs_5_8_0.01_48_5_0_few_shot


