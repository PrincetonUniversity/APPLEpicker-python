#!/bin/bash

# This file is an execution file for CodeOcean cloud platform
# It allows you to run the code without diving into the cumbersome process of installing and managing a
# virtual environment, pipenv dependencies etc.
# You can view a live example at https://codeocean.com/capsule/4705dc64-1815-48ec-882c-a803aea53908/code

set -e  # stop on error

# verify we're running on CodeOcean
if [ ! -f /.dockerenv ]; then
        echo "This script is meant to run only on Code-Ocean platform."
        echo "Please check https://codeocean.com/capsule/4705dc64-1815-48ec-882c-a803aea53908/code."
        exit
fi

echo "running ApplePicker.."
python3 run.py  -s 78  --jpg  -o /results  /data

echo "Done. You can check the results in the right column."

