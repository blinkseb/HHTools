#! /bin/bash

if [ ! -d "keras" ]; then
    echo "Keras env. is not setup. Running setup.sh for you"
    set +e
    ./keras/setup.sh
    set -e
    echo "Done"
fi

# FIXME: Detect that the user actually source the file

module load gcc/gcc-4.9.1-sl6_amd64
module load python/python27_sl6_gcc49

if [ -z "$VIRTUAL_ENV" ]; then
    source keras/bin/activate
fi

# Load ROOT
source /home/fynu/sbrochet/HH/root-6-08-00/bin/thisroot.sh

export OMP_NUM_THREADS=8
