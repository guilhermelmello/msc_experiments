#PBS -N model_selection
#PBS -q testegpu
#PBS -e outputs/model_selection.err
#PBS -o outputs/model_selection.out


# load python and tensorflow
module load tensorflow/2.6.0-gcc-9.3.0
module load python/3.8.11-gcc-9.4.0


# projec dir
cd ~/msc_experiments
source ~/venvs/msc_env/bin/activate

echo "Using as root directory:"
pwd


# set HF to offline mode
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


# gpu usage
echo "Setting GPU monitor"
nvidia-smi pmon >> outputs/usogpu &

# run python script
echo "Running python script"
python -m scripts.model_selection


# unload virtualenv
echo "Deactivating virtual environment"
deactivate
