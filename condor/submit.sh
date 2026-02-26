#!/bin/bash -xe

JOBID=$1
BASEDIR=$2
X509PATH=$3

export HOME=`pwd`
if [ -d /afs/cern.ch/user/${USER:0:1}/$USER ]; then
  export HOME=/afs/cern.ch/user/${USER:0:1}/$USER
fi

export XRD_NETWORKSTACK=IPv4
export XRD_RUNFORKHANDLER=1
export X509_USER_PROXY=$X509PATH
voms-proxy-info -all -file $X509PATH

WORKDIR=`pwd`

declare -A ARGS
for key in workflow year output_path output_format dataset eos user use_6class mva_model_path use_pytorch mva_hidden_dim mva_num_layers model_type mass_window; do
    ARGS[$key]=$(python3 -c "import json; print(json.load(open('$WORKDIR/arguments.json')).get('$key', 'None'))")
done

OPTS="--workflow ${ARGS[workflow]} --year ${ARGS[year]} --output_path ${ARGS[output_path]} --output_format ${ARGS[output_format]} --dataset ${ARGS[dataset]}_$JOBID --user ${ARGS[user]}"

if [ "${ARGS[eos]}" = "True" ]; then
    OPTS="$OPTS --eos"
fi

if [ "${ARGS[use_6class]}" = "True" ]; then
    OPTS="$OPTS --use_6class"
fi

if [ "${ARGS[mva_model_path]}" != "None" ] && [ -n "${ARGS[mva_model_path]}" ]; then
    OPTS="$OPTS --mva_model_path ${ARGS[mva_model_path]}"
fi

if [ "${ARGS[use_pytorch]}" = "True" ]; then
    OPTS="$OPTS --use_pytorch"
    OPTS="$OPTS --mva_hidden_dim ${ARGS[mva_hidden_dim]}"
    OPTS="$OPTS --mva_num_layers ${ARGS[mva_num_layers]}"
    if [ "${ARGS[model_type]}" != "None" ] && [ -n "${ARGS[model_type]}" ]; then
        OPTS="$OPTS --model_type ${ARGS[model_type]}"
    fi
fi

if [ "${ARGS[mass_window]}" = "True" ]; then
    OPTS="$OPTS --mass_window"
fi

# get partition fileset
python3 -c "import json; json.dump(json.load(open('$WORKDIR/partitions.json'))['$JOBID'], open('$WORKDIR/partition_fileset.json', 'w'), indent=4)"
OPTS="$OPTS --partition_json $WORKDIR/partition_fileset.json"

echo $OPTS

cd $BASEDIR

# Install dependencies for MVA inference if needed
if [ "${ARGS[mva_model_path]}" != "None" ] && [ -n "${ARGS[mva_model_path]}" ]; then
    if [ "${ARGS[use_pytorch]}" = "True" ]; then
        pip install --user torch --index-url https://download.pytorch.org/whl/cpu
    else
        pip install --user onnxruntime
    fi
fi

python3 submit.py $OPTS