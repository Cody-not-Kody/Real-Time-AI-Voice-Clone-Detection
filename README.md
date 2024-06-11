# Real-Time-AI-Voice-Clone-Detection
A CNN-GRU neural network approach for a real time AI voice clone detection. This was my Master's Capstone project and accepted as the proceedings to the 2024 International Conference on Data Science. 

This repository contains the jupyter notebook and python file used to train and test detection models. The repository also contains an AI-generated voice dataset which are cloned samples of the [LJ Voice Dataset](https://keithito.com/LJ-Speech-Dataset/). 

Two types of AI voice samples were curated, one using [Speechify Voice Cloning](https://speechify.com/voice-cloning/) and another using [Jarod Mica's AI Voice Cloning Tool](https://github.com/JarodMica/ai-voice-cloning)


## Requirements
1. Python 3.9 is recommended using Pytorch-cuda 11.8. GPU is recommended but is not required.
2. A metadata file outlining the class label for each sample is needed for each time a dataloader is initialized. A jupyter notebook is provided to do this but will need to adjusted per your needs. An example of a metadata file is provided in the Datasets directory and will need to follow that structure. 
3. (Optional) You will need to download the [LJ Voice Dataset](https://keithito.com/LJ-Speech-Dataset/) for legitimate voice samples when testing with the curated AI voice dataset. 
