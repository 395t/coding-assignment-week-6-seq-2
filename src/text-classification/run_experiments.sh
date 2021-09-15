# I disable the training (--do_train=False, zero-shot evaluation) in this script.

for TASK_NAME in cola sst2 mrpc stsb qqp mnli qnli rte wnli
do
    echo $TASK_NAME
    CUDA_VISIBLE_DEVICES=0 python run_glue.py \
        --model_name_or_path $MODEL_NAME \
        --task_name $TASK_NAME \
        --do_eval \
        --max_seq_length 128 \
        --per_device_train_batch_size 32 \
        --learning_rate 2e-5 \
        --num_train_epochs 3 \
        --output_dir saved_dir/${MODEL_NAME}/${TASK_NAME}/
done
