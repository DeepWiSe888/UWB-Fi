# UWB-Fi: Pushing Wi-Fi towards Ultra-wideband for Fine-Granularity Sensing

Welcome to UWB-Fi! This repository contains the code and resources to implement UWB-Fi.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Preprocessing Data](#preprocessing-data)
- [Running UWB-Fi's SpecTrans Model](#running-uwb-fis-spectrans-model)
- [Postprocessing Data](#postprocessing-data)

## Introduction

UWB-Fi is the first Wi-Fi system to achieve physical UWB sensing using only discrete and irregular channel samples across nearly 5GHz bandwidth.


## Getting Started

To begin using UWB-Fi and implement the system, follow these steps:

1. **Clone the Repository**: Start by cloning this repository to your local machine using the following command:

   ```bash
   git clone https://github.com/DeepWiSe888/UWB-Fi
   ```

2. **Install Python 3.8**: Ensure Python 3.8 is installed on your computer. If not, download the latest version from the official website: [python.org](https://www.python.org/).

3. **Set Up Virtual Environment**: It's recommended to set up a virtual environment to maintain a clean and isolated environment for your UWB-Fi implementation. Use tools like `virtualenv` or `conda` for this purpose. Activate your virtual environment before proceeding.

4. **Install the necessary packages** such as pytorch 2.0.1 (find more details in requirements.txt in the `UWB_Fi_python` folder).

5. **Download Datasets**: The specific datasets used with UWB-Fi can be obtained from [Dataset Download](https://entuedu-my.sharepoint.com/:u:/g/personal/hongbo001_e_ntu_edu_sg/EQMBtZTgYf1CqLQ9AhMu0k8B3gljzw3jGuf1VvcjiN43UQ?e=MWtEku). Extract the contents of the `dataset_mat.zip` file and  make sure the extracted `dataset_mat` folder is placed into the `UWB_Fi_matlab` folder.
Note that the data dimension of locx.mat is (20, 57, 4, 210), where 20 represents the 20 channels, 57 represents the number of subcarriers, 4 is the result of 2 (the number of antennas) multiplied by 2 (the real and imaginary parts of CSI), and 210 represents the number of samples. The data can be collected by [Picoscenes](https://ps.zpj.io/). The `PicoScenes-PDK-modified` folder is a simplified version of the plugin for Linux kernel version 5.15.0-60, available for user testing. We will upload the new version of PicoScenes for Linux kernel version 5.15.0-78 at a later time. If users need to conduct tests on the new version of PicoScenes, they can temporarily use the existing frequency sweeping plugin command as a substitute.

Now you're ready to implement UWB-Fi!

## Preprocessing Data

The `UWB_Fi_matlab` folder contains the Matlab code for preprocessing data and postprocessing the data obtained after the neural network. To preprocess the data, follow these steps: 

1. **Navigate to the `UWB_Fi_matlab` folder**:

   ```bash
   cd UWB_Fi_matlab
   ```

2. **Run the `main_pre.m` Script**: Execute the `main_pre.m` script using MATLAB or appropriate software.

3. **Move Preprocessed Data**: Transfer the produced `traindata.mat` and `testdata.mat` files from the `UWB_Fi_matlab/dataset_for_model` folder to the `UWB_Fi_python/data` folder.

## Running UWB-Fi's SpecTrans Model

The `UWB_Fi_python` folder contains the Python code for the SpecTrans Model of UWB-Fi. To run the code, follow these steps:

1. **Navigate to the `UWB_Fi_python` folder**:

   ```bash
   cd UWB_Fi_python
   ```

2. **Activate Virtual Environment**: Ensure your virtual environment is activated.

3. **Run the `train_det.py` Script**: Execute the `train_det.py` script using Python:

   ```bash
   python train_det.py
   ```

4. **Transfer Results**: Transfer the produced `pre_result.mat` file from the `UWB_Fi_python/experiments/deterministic/pre_result/predictions` folder to the `UWB_Fi_matlab/SpecTrans_result` folder.

## Postprocessing Data

After obtaining results from the SpecTrans Model, postprocess the data using MATLAB. Follow these steps: 

1. **Navigate to the `UWB_Fi_matlab` folder**:

   ```bash
   cd UWB_Fi_matlab
   ```

2. **Run the `post_process_data.m` Script**: Execute the `post_process_data.m` script using MATLAB or appropriate software.



