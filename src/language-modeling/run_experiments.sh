export WANDB_PROJECT=week6-language-modeling

CUDA_VISIBLE_DEVICES=0 python run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --report_to wandb \
    --output_dir save_dir/gpt2-clm &

CUDA_VISIBLE_DEVICES=1 python run_mlm.py \
    --model_name_or_path roberta-base \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --report_to wandb \
    --output_dir save_dir/roberta_base &

CUDA_VISIBLE_DEVICES=2 python run_plm.py \
    --model_name_or_path=xlnet-base-cased \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --report_to wandb \
    --output_dir save_dir/xlnet &

CUDA_VISIBLE_DEVICES=3 python run_mlm.py \
    --model_name_or_path roberta-large \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --report_to wandb \
    --output_dir save_dir/roberta_large &

wait
