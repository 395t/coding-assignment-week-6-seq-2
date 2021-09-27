# coding-template

## Summary

The summary can contain but is not limited to:

- Code structure.

- In order to run out tests, we had to upload the `run_glue.py` into our Google Colab Notebook. Then, the test script that we used to run tests back-to-back, is th following:
(In Google Colab) This can be run on a GPU by switching the Google Colab runtime to GPU.

```
!pip install datasets
!pip install git+https://github.com/huggingface/transformers

...

%%shell
for MODEL_NAME in albert-base-v1 bert-base-cased bert-large-case xlnet-base-cased xlnet-large-case roberta-base
do
    for TASK_NAME in qqp mnli qnli rte wnli
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
            --output_dir drive/MyDrive/COLAB_OUTPUTS/saved_dir/${MODEL_NAME}/${TASK_NAME}/
    done
done
```

- Write-up of your findings and conclusions.

<img src='./imgs/ALBERT_BASE_ACCURACY.png' width="40%">

<img src='./imgs/ALBERT_BASE_LOSS.png' width="40%">

<img src='./imgs/BERT_BASE_ACCURACY.png' width="40%">

<img src='./imgs/BERT_BASE_LOSS.png' width="40%">

<img src='./imgs/BERT_BASE_RUNTIME.png' width="40%">

<img src='./imgs/BERT_BASE_SAMPLES.png' width="40%">

<img src='./imgs/BERT_BASE_STEPS.png' width="40%">

<img src='./imgs/BERT_LARGE_ACCURACY.png' width="40%">

<img src='./imgs/BERT_LARGE_LOSS.png' width="40%">

<img src='./imgs/ROBERTA_ACCURACY.png' width="40%">

<img src='./imgs/ROBERTA_LOSS.png' width="40%">

<img src='./imgs/XLNET_ACCURACY.png' width="40%">

<img src='./imgs/XLNET_LOSS.png' width="40%">

<img src='./imgs/XLNET_LARGE_ACCURACY.png' width="40%">

<img src='./imgs/XLNET_LARGE_LOSS.png' width="40%">

<img src='./imgs/PARAMETERS.png' width="40%">

- Ipython notebooks can be organized in `notebooks`.

## Reference

Any code that you borrow or other reference should be properly cited.
