export PYTHONPATH=.
DEVICE=0;

# data
CONFIG="configs/exp/affectflow_dailytalk_bert.yaml";
binary_data_dir="/dataset/binary/dailytalk_22.5khz";

# code 
task_class="tasks.tts.AffectFlow.ExpressiveFS2Task";
dataset_class="tasks.tts.dataset_utils.AffectFlow2Dataset";
model_class="models.tts.AffectFlow.ExpressiveFS2";

run() {
    local MODEL_NAME=$1
    local FINAL_MODEL_NAME=${MODEL_NAME}
    local GEN_DIR=./results/${MODEL_NAME}/generated_160000_${config_suffix}/wavs
    local FINAL_GEN_DIR=./results/${MODEL_NAME}/generated_160000_${config_suffix}/final_wavs

    echo "config_suffix: $config_suffix"
    local HPARAMS="binary_data_dir=$binary_data_dir,task_cls=$task_class,dataset_cls=$dataset_class,model_cls=$model_class,gen_dir_name=$config_suffix"

    # Train
    CUDA_VISIBLE_DEVICES=$DEVICE python tasks/run.py \
        --config $CONFIG \
        --exp_name $MODEL_NAME \
        --hparams=$HPARAMS \
        --reset

    # Infer
    CUDA_VISIBLE_DEVICES=$DEVICE python tasks/run.py \
        --config $CONFIG \
        --exp_name $MODEL_NAME \
        --infer \
        --hparams=$HPARAMS \
        --reset
}

#########################
#   Run for the model   #
#########################
run "AffectFlow"