#declare -a arr=("save_dir/gpt2-large-clm_1e-5_gradacc1_4nodes/checkpoint-500" "gpt2-large" \
#    "save_dir/gpt2-medium-clm_2e-5_gradacc2_4nodes/checkpoint-500" "gpt2-medium" \
#    "save_dir/gpt2-clm/checkpoint-500" "gpt2")
#for model_name in "${arr[@]}"
#do
#    echo $model_name
#    CUDA_VISIBLE_DEVICES=0 python run_generation.py \
#        --prompt="Who is behind the 9/11 event?" \
#        --length=512 \
#        --fp16 \
#        --model_type=gpt2 \
#        --model_name_or_path=$model_name
#done

declare -a arr=("save_dir/gptneo1.3B/checkpoint-500" "EleutherAI/gpt-neo-1.3B" \
    "save_dir/gptneo125M/checkpoint-400" "EleutherAI/gpt-neo-125M")
for model_name in "${arr[@]}"
do
    echo $model_name
    CUDA_VISIBLE_DEVICES=0 python run_generation.py \
        --prompt="Who is behind the 9/11 event?" \
        --length=512 \
        --fp16 \
        --model_type=gpt-neo \
        --model_name_or_path=$model_name
done
