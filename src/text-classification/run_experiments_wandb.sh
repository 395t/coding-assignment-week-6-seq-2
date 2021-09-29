# I disable the training (--do_train=False, zero-shot evaluation) in this script.
# This script requires wandb setup

export WANDB_PROJECT=week6-text-classification
#MODEL_NAME=bert-large-cased

for MODEL_NAME in xlnet-large-cased
do
    for TASK_NAME in sst2
    do 
        echo $TASK_NAME
        CUDA_VISIBLE_DEVICES=3 python run_glue.py \
            --model_name_or_path $MODEL_NAME \
            --task_name $TASK_NAME \
            --do_train \
            --do_eval \
            --max_seq_length 128 \
            --per_device_train_batch_size 16 \
            --learning_rate 1e-5 \
            --num_train_epochs 5 \
            --report_to wandb \
            --output_dir saved_dir_trained/${MODEL_NAME}/${TASK_NAME}/
    done
done

#for MODEL_NAME in bert-large-cased albert-base-v1 xlnet-base-cased roberta-base
#for MODEL_NAME in bert-large-cased
#do
#    for TASK_NAME in sst2 mrpc qqp mnli stsb qnli rte wnli cola
#    do 
#        echo $TASK_NAME
#        CUDA_VISIBLE_DEVICES=0 python run_glue.py \
#            --model_name_or_path $MODEL_NAME \
#            --task_name $TASK_NAME \
#            --do_train \
#            --do_eval \
#            --max_seq_length 128 \
#            --per_device_train_batch_size 32 \
#            --learning_rate 2e-5 \
#            --num_train_epochs 5 \
#            --report_to wandb \
#            --output_dir trained_saved_dir/${MODEL_NAME}/${TASK_NAME}/
#    done
#done

#for MODEL_NAME in roberta-base 
#do
#    for TASK_NAME in sst2 mrpc qqp mnli stsb qnli rte wnli cola
#    do 
#        echo $TASK_NAME
#        CUDA_VISIBLE_DEVICES=1 python run_glue.py \
#            --model_name_or_path $MODEL_NAME \
#            --task_name $TASK_NAME \
#            --do_train \
#            --do_eval \
#            --max_seq_length 128 \
#            --per_device_train_batch_size 32 \
#            --learning_rate 2e-5 \
#            --num_train_epochs 5 \
#            --report_to wandb \
#            --output_dir trained_saved_dir/${MODEL_NAME}/${TASK_NAME}/
#    done
#done

#for MODEL_NAME in bert-large-cased
#do
#    for TASK_NAME in sst2 mrpc qqp mnli stsb qnli rte wnli cola
#    do 
#        echo $TASK_NAME
#        CUDA_VISIBLE_DEVICES=2 python run_glue.py \
#            --model_name_or_path $MODEL_NAME \
#            --task_name $TASK_NAME \
#            --do_train \
#            --do_eval \
#            --max_seq_length 128 \
#            --per_device_train_batch_size 32 \
#            --learning_rate 2e-5 \
#            --num_train_epochs 5 \
#            --report_to wandb \
#            --fp16 \
#            --output_dir saved_dir_trainedfp16/${MODEL_NAME}/${TASK_NAME}/
#    done
#done

#for MODEL_NAME in xlnet-large-cased
#do
#    for TASK_NAME in sst2 mrpc qqp mnli stsb qnli rte wnli cola
#    do 
#        echo $TASK_NAME
#        CUDA_VISIBLE_DEVICES=3 python run_glue.py \
#            --model_name_or_path $MODEL_NAME \
#            --task_name $TASK_NAME \
#            --do_train \
#            --do_eval \
#            --max_seq_length 128 \
#            --per_device_train_batch_size 16 \
#            --learning_rate 2e-5 \
#            --num_train_epochs 5 \
#            --report_to wandb \
#            --output_dir saved_dir_trained/${MODEL_NAME}/${TASK_NAME}/
#    done
#done
