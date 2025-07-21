#!/bin/bash

# Activate model environment
source venv_model/bin/activate

# Start model training
cd dressing-in-order-main
python train.py