# DPFedMeta
Code for paper 'Squeezing More Utility via Adaptive Clipping on Deferentially Private Gradients in Federated Meta-Learning' ACSAC 2022

## The running platform
- **Hardware:** a server equipped with a 3.3 GHz Intel Core i9-9820X CPU, three GeForce RTX 2080 Ti GPUs (a different CPU/GPU is ok if it support the Pytorch deep learning framewotk)
- **Operating system:** Ubuntu 18.04.3 LTS (a different version of Ubuntu is ok if it support the Pytorch deep learning framewotk)
- **Deep learning framework:** Pytorch 1.4.0
- **Language:** Python 3.6.10

See "**Prepare the environment**" to set up Pytorch 1.4.0 and Python 3.6.10. Please read step by step to get ready for running the code.

## Setup GPU for deep learning
If you have already set up your GPU for deep learning, please ignore this step and go to "Prepare the environment".

If you have never setup your GPU machine for deep learning, a guide is available at [Deep Learning GPU Installation](https://towardsdatascience.com/deep-learning-gpu-installation-on-ubuntu-18-4-9b12230a1d31).
This guide includes three steps: NVIDIA Driver installation, CUDA installation, and CUDNN installation. 


## Prepare the environment
1. Implement annoconda following the instruction in https://www.anaconda.com/. Anaconda will help to manage the learning environement.
2.  Create your env using the *environment.yml* file included in the repository. All the required libraries (including pytorch and python) will be implemented. The default environment name is *myenv* as specified in the yml file. If you would like to use another name, edit the first line of the *environment.yml* file.
-      conda env create --file environment.yml
3. activate your environment. 
-      conda activate envname


## Download the source code
Download the repository to your local machine
-      git clone git@github.com:ning-wang1/DPFedMeta.git
The downloaded folder is named as 'DPFedMeta', I'll call this folder by the project folder in the following. The folder structure is:

```bash
├── checkpoint
├── experiment_config
├── experiment_scripts
├── result
├── tfcode
│   ├── __init__.py
│   └── rdp_accountant.py
├── utils
│   ├── __init__py
│   ├── dataset_tools.py
│   ├── dp_utils.py
│   ├── mnist.py
│   ├── parser_utils.py
│   └── storage.py
├── .gitignore
├── base_experiment.py
├── compute_dp_sgd_privacy.py
├── data.py
├── data_tasks_split.py
├── df_few_shot_learning_system.py
├── df_few_shot_learning_system_example_level_single.py
├── DPAGR.py
├── DPAGRLR.py
├── environment.yml
├── experiment_builder.py
├── inner_loop_optimizer.py
├── meta_neural_network_architectures.py
├── README.md
└── vgg.py
```
The main functions are DPAGR.py and DPAGRLR.py

## Download the datasets
We use three datasets, including Omniglot [1], CIFAR-FS [2], and Mini-ImageNet [3]. For convenience, the link to download the data are listed:

[omniglot dataset](https://www.omniglot.com/)

[CIFAR-FS dataset](https://drive.google.com/file/d/1pTsCCMDj45kzFYgrnO67BWVbKs48Q3NI/view)

[mini-ImageNet dataset](https://drive.google.com/file/d/1R6dA6QGEW-lmiNkitCwK4IkAbl4uT3y3/view)

1. Create a folder in your project folder and name it as 'datasets'
2. Download the datasets from the above links
3. Rename each download dataset in case its default name is different from that used in the code.
    1. name the ominiglot dataset as *'omniglot_dataset'*
    2. name the CIFAR-FS dataset as *'cifar_100'*
    3. name the mini-ImageNet dataset as *'mini_imagenet_full_size'*
Be sure to coordnate your reference path and the dataset path.


[1] Brenden M Lake et al. 2015. Human-level concept learning through probabilistic program induction. Science 350, 6266 (2015), 1332–1338

[2] Luca Bertinetto, Joao F Henriques, Philip HS Torr, and Andrea Vedaldi. 2019. Meta-learning with differentiable closed-form solvers. In International Conference on Learning Representations (ICLR 19)

[3] Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Daan Wierstra, et al . 2016. Matching networks for one shot learning. In Advances in neural information processing systems (NeurIPS 16). 3630–3638

## Run the experiment 
1. go to the folder experiment_scripts: cd experiment_scripts
2. Training with DPAGR algorithm that protect user level privacy
      - Omniglot dataset: bash omniglot_DPAGR.sh
      - mini_ImageNet dataset: bash mini-imagenet_DPAGR.sh 
      - CIFAR-FS dataset: bash cifar-fs_DPAGR.sh

        ii. Training with DPAGRLR algorithm that protect both user level privacy and record level privacy
        
-      Omniglot dataset: bash omniglot_DPAGRLR.sh
-      mini_ImageNet dataset: bash mini-imagenet_DPAGRLR.sh 
-      CIFAR-FS dataset: bash cifar-fs_DPAGRLR.sh



