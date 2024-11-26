# Session 3
Train a model remotely using Google Cloud Compute Engine.
## Installation
### With Conda
Create a conda environment by running
```
conda create --name aidl-session3 python=3.9
```
Then, activate the environment
```
conda activate aidl-session3
```
and install the dependencies
```
pip install -r requirements.txt
```
## Running the project

To run the project without hyperparameter tuning, run
```
python session-3/main.py
```

# Car vs Flower Classifier

A deep learning model that classifies images as either cars or flowers using PyTorch.

## Project Structure
- `main.py`: Training script
- `model.py`: Model architecture
- `utils.py`: Utility functions
- `predict.py`: Inference script for testing new images

## Setup
```bash
pip install torch torchvision
```

## for github config
First, create a Personal Access Token on GitHub:

Go to GitHub.com and log in
Click your profile picture → Settings
Scroll down to Developer Settings (bottom of left sidebar)
Click Personal Access Tokens → Tokens (classic)
Generate New Token (classic)
Give it a name (like "Command line access")
Select scopes: check at least repo
Click "Generate token"
IMPORTANT: Copy the token immediately - you won't be able to see it again

