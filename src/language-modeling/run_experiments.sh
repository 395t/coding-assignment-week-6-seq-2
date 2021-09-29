export WANDB_PROJECT=week6-language-modeling

#CUDA_VISIBLE_DEVICES=0 python run_clm.py \
#python -m torch.distributed.launch \
#    --nproc_per_node=4 \
#deepspeed run_clm.py \
CUDA_VISIBLE_DEVICES=0 python run_clm.py \
    --model_name_or_path gpt2-medium \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --learning_rate 1e-4 \
    --fp16 \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 15 \
    --logging_steps 25 \
    --eval_steps 10 \
    --output_dir save_dir/gpt2-medium-test
    #--deepspeed deepspeed_config.json \
    #--report_to wandb \

#CUDA_VISIBLE_DEVICES=1 python run_mlm.py \
#python -m torch.distributed.launch \
#    --nproc_per_node=4 \
#    run_mlm.py \
#    --model_name_or_path roberta-base \
#    --dataset_name wikitext \
#    --dataset_config_name wikitext-2-raw-v1 \
#    --do_train \
#    --do_eval \
#    --report_to wandb \
#    --deepspeed deepspeed_config.json \
#    --learning_rate 1e-4 \
#    --fp16 \
#    --per_device_train_batch_size 4 \
#    --per_device_eval_batch_size 1 \
#    --num_train_epochs 15 \
#    --logging_steps 25 \
#    --eval_steps 400 \
#    --output_dir save_dir/roberta_base

#python -m torch.distributed.launch \
#    --nproc_per_node=4 \
#deepspeed run_plm.py \
#    --model_name_or_path=xlnet-base-cased \
#    --dataset_name wikitext \
#    --dataset_config_name wikitext-2-raw-v1 \
#    --do_train \
#    --do_eval \
#    --report_to wandb \
#    --deepspeed deepspeed_config.json \
#    --learning_rate 1e-4 \
#    --fp16 \
#    --per_device_train_batch_size 4 \
#    --per_device_eval_batch_size 1 \
#    --num_train_epochs 15 \
#    --logging_steps 25 \
#    --eval_steps 400 \
#    --output_dir save_dir/xlnet

#python -m torch.distributed.launch \
#    --nproc_per_node=4 \
#deepspeed run_mlm.py \
#    --model_name_or_path roberta-large \
#    --dataset_name wikitext \
#    --dataset_config_name wikitext-2-raw-v1 \
#    --do_train \
#    --do_eval \
#    --report_to wandb \
#    --deepspeed deepspeed_config.json \
#    --learning_rate 1e-4 \
#    --fp16 \
#    --per_device_train_batch_size 4 \
#    --per_device_eval_batch_size 1 \
#    --num_train_epochs 15 \
#    --logging_steps 25 \
#    --eval_steps 400 \
#    --output_dir save_dir/roberta_large
