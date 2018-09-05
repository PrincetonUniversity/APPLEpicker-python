#!/bin/bash

# This file is an execution file for CodeOcean cloud platform
# It allows you to run the code without diving into the cumbersome process of installing and managing a
# virtual environment, pipenv dependencies etc.
# You can view a live example at https://codeocean.com/capsule/4705dc64-1815-48ec-882c-a803aea53908/code

set -e  # stop on error

# verify we're running on Code Ocean
if [ `printenv PWD` != "/code" ]
then
        echo "This script is meant to run only on Code-Ocean platform."
        echo "Please check https://codeocean.com/capsule/4705dc64-1815-48ec-882c-a803aea53908/code."
        exit
fi

echo "installing pipenv.."
pip3 install pipenv

echo "creating a virtual environment for Python3"
pipenv --python 3.5

echo "installing python packages (dependencies) using pipenv.."
pipenv install

echo "running APPLE picker.."
pipenv run python3 apple.py  -s 78  -o ../results/  ../../input

echo "Done. You can check the results in the right column."
