unset LC_ALL

MAIN_ROOT=$PWD/espnet
KALDI_ROOT=$MAIN_ROOT/tools/kaldi
SPNET_ROOT=$MAIN_ROOT/src

export PATH=$PWD/python:$PWD/shell:$SPNET_ROOT/utils/:$SPNET_ROOT/bin/:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$MAIN_ROOT/tools/chainer_ctc/ext/warp-ctc/build
source $MAIN_ROOT/tools/venv/bin/activate
export PYTHONPATH=$PWD/python:$SPNET_ROOT/lm/:$SPNET_ROOT/asr/:$SPNET_ROOT/nets/:$SPNET_ROOT/utils/:$SPNET_ROOT/bin/:$PYTHONPATH

export OMP_NUM_THREADS=1
