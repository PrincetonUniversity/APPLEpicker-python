#!/bin/bash

# This file is an execution file for CodeOcean cloud platform
# It allows you to run the code without diving into the cumbersome process of installing and managing a
# virtual environment, pipenv dependencies etc.
# You can view a live example at https://codeocean.com/capsule/3477951b-31c5-4e45-8894-99e256fc5836/code

set -e
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

echo "upgrading pip.."
pip install --upgrade pip

echo "installing virtual environment.."
pip3 install -U virtualenv

echo "creating virtualenv.."
virtualenv --python=python3.5 ve-ap

echo "activating virtualenv.."
source ve-ap/bin/activate

echo "installing pipenv.."
pip install pipenv

echo "installing python packages (dependencies) using pipenv.."
pipenv install

echo "running APPLE picker.."
python apple.py  -s 78  -o ../results/  ../../input

echo "done. you can check the results in the right column."
