#!/bin/bash

# untar your Python installation. Make sure you are using the right version!
tar -xzf python39.tar.gz
# (optional) if you have a set of packages (created in Part 1), untar them also
tar -xzf packages.tar.gz

tar -xvf data.tar


# make sure the script will use your Python installation, 
# and the working directory as its home location
export PATH=$PWD/python/bin:$PATH
export PYTHONPATH=$PWD/packages
export HOME=$PWD

# run your script
python3 model_train.py $@
