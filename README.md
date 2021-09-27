# Text-classification 

## Summary

The summary can contain but is not limited to:

- Code structure.
### Data Sets Summary

#### Glue

GLUE, the General Language Understanding Evaluation benchmark (https://gluebenchmark.com/) is a collection of resources for training, evaluating, and analyzing natural language understanding systems.

#### cola

- **Config description**: The Corpus of Linguistic Acceptability consists of English acceptability judgments drawn from books and journal articles on linguistic theory. Each example is a sequence of words annotated with whether it is a grammatical English sentence.

- **Dataset size**: 965.49 KiB

- **Examples**:
<p align="center">
<img src='./imgs/COLA_example.png' width="100%">
</p>

#### sst2

- **Config description**: The Stanford Sentiment Treebank consists of sentences from movie reviews and human annotations of their sentiment. The task is to predict the sentiment of a given sentence. We use the two-way (positive/negative) class split, and use only sentence-level labels.

- **Dataset size**: 7.22 MiB

- **Examples**:
<p align="center">
<img src='./imgs/SST2_example.png' width="100%">
</p>

#### mrpc

- **Config description**: The Microsoft Research Paraphrase Corpus (Dolan & Brockett, 2005) is a corpus of sentence pairs automatically extracted from online news sources, with human annotations for whether the sentences in the pair are semantically equivalent.

- **Dataset size**: 1.74 MiB

- **Examples**:
<p align="center">
<img src='./imgs/mrpc_example.png' width="100%">
</p>

#### stsb

- **Config description**: The Semantic Textual Similarity Benchmark (Cer et al., 2017) is a collection of sentence pairs drawn from news headlines, video and image captions, and natural language inference data. Each pair is human-annotated with a similarity score from 1 to 5.

- **Dataset size**: 1.58 MiB

- **Examples**:
<p align="center">
<img src='./imgs/stsb_example.png' width="100%">
</p>

#### qqp

- **Config description**: The Quora Question Pairs2 dataset is a collection of question pairs from the community question-answering website Quora. The task is to determine whether a pair of questions are semantically equivalent.

- **Dataset size**: 150.37 MiB

- **Examples**:
<p align="center">
<img src='./imgs/qqp_example.png' width="100%">
</p>

#### mnli

- **Config description**: The Multi-Genre Natural Language Inference Corpus is a crowdsourced collection of sentence pairs with textual entailment annotations. Given a premise sentence and a hypothesis sentence, the task is to predict whether the premise entails the hypothesis (entailment), contradicts the hypothesis (contradiction), or neither (neutral). The premise sentences are gathered from ten different sources, including transcribed speech, fiction, and government reports. We use the standard test set, for which we obtained private labels from the authors, and evaluate on both the matched (in-domain) and mismatched (cross-domain) section. We also use and recommend the SNLI corpus as 550k examples of auxiliary training data.

- **Dataset size**: 100.56 MiB

- **Examples**:
<p align="center">
<img src='./imgs/mnli_example.png' width="100%">
</p>

#### qnli

- **Config description**: The Stanford Question Answering Dataset is a question-answering dataset consisting of question-paragraph pairs, where one of the sentences in the paragraph (drawn from Wikipedia) contains the answer to the corresponding question (written by an annotator). We convert the task into sentence pair classification by forming a pair between each question and each sentence in the corresponding context, and filtering out pairs with low lexical overlap between the question and the context sentence. The task is to determine whether the context sentence contains the answer to the question. This modified version of the original task removes the requirement that the model select the exact answer, but also removes the simplifying assumptions that the answer is always present in the input and that lexical overlap is a reliable cue.

- **Dataset size**: 32.99 MiB

- **Examples**:
<p align="center">
<img src='./imgs/qnli_example.png' width="100%">
</p>

#### rte

- **Config description**: The Recognizing Textual Entailment (RTE) datasets come from a series of annual textual entailment challenges. We combine the data from RTE1 (Dagan et al., 2006), RTE2 (Bar Haim et al., 2006), RTE3 (Giampiccolo et al., 2007), and RTE5 (Bentivogli et al., 2009).4 Examples are constructed based on news and Wikipedia text. We convert all datasets to a two-class split, where for three-class datasets we collapse neutral and contradiction into not entailment, for consistency.

- **Dataset size**: 2.15 MiB

- **Examples**:
<p align="center">
<img src='./imgs/rte_example.png' width="100%">
</p>

#### wnli

- **Config description**: The Winograd Schema Challenge (Levesque et al., 2011) is a reading comprehension task in which a system must read a sentence with a pronoun and select the referent of that pronoun from a list of choices. The examples are manually constructed to foil simple statistical methods: Each one is contingent on contextual information provided by a single word or phrase in the sentence. To convert the problem into sentence pair classification, we construct sentence pairs by replacing the ambiguous pronoun with each possible referent. The task is to predict if the sentence with the pronoun substituted is entailed by the original sentence. We use a small evaluation set consisting of new examples derived from fiction books that was shared privately by the authors of the original corpus. While the included training set is balanced between two classes, the test set is imbalanced between them (65% not entailment). Also, due to a data quirk, the development set is adversarial: hypotheses are sometimes shared between training and development examples, so if a model memorizes the training examples, they will predict the wrong label on corresponding development set example. As with QNLI, each example is evaluated separately, so there is not a systematic correspondence between a model's score on this task and its score on the unconverted original task. We call converted dataset WNLI (Winograd NLI).

- **Dataset size**: 198.88 KiB

- **Examples**:
<p align="center">
<img src='./imgs/wnli_example.png' width="100%">
</p>

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

## Models

### XLNet

We used two types of architecture, [base cased](https://huggingface.co/xlnet-base-cased) a 12-layer, 768-hidden, 12-heads and a [large cased](https://huggingface.co/xlnet-large-cased)  24-layer, 1024-hidden, 16-heads evaluated on the nine  text classification tasks.Based on [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) paper implemented by hugging-face.
### Roberta-Base

We used the a 125M parameter pre-trained [roberta-base](https://huggingface.co/roberta-base) model card implemented by hugging-face based on the [RoBERTa: A Robustly Optimized BERT Pretraining Approach paper ](https://arxiv.org/abs/1907.11692).Model pre-trained using only masked language modeling (MLM) on raw texts with automatic labelling.


### Bert-Large

We used a 24-layer, 1024-hidden, 16-heads, 340M parameters  pre-trained [bert-uncased](https://huggingface.co/bert-base-uncased) model card implemented by hugging-face, based on the [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding paper](https://arxiv.org/abs/1810.04805). Model was pre-trained using both masked language modeling (MLM) and next sentence prediction (NSP),on large raw texts without human labelling.


### Albert-Base

We used a 12-layer, 768-hidden, 12M parameters  pre-trained uncased model card implemented [Albert-base](https://huggingface.co/transformers/model_doc/albert.html) that is base on the [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations paper](https://arxiv.org/abs/1909.11942).Model was pre-trained using both masked language modelling (MLM) and sentence ordering prediction (SOP) on a large corpus of English data on raw texts only without humans labels,labels are generated automatically texts.

<!-- <img src='./imgs/ALBERT_BASE_ACCURACY.png' width="40%"> -->

<!-- <img src='./imgs/ALBERT_BASE_LOSS.png' width="40%"> -->

<!-- <img src='./imgs/BERT_BASE_ACCURACY.png' width="40%"> -->

<!-- <img src='./imgs/BERT_BASE_LOSS.png' width="40%"> -->

<img src='./imgs/BERT_BASE_RUNTIME.png' width="40%">

<img src='./imgs/BERT_BASE_SAMPLES.png' width="40%">

<img src='./imgs/BERT_BASE_STEPS.png' width="40%">

<!-- <img src='./imgs/BERT_LARGE_ACCURACY.png' width="40%"> 

<img src='./imgs/BERT_LARGE_LOSS.png' width="40%"> 

<img src='./imgs/ROBERTA_ACCURACY.png' width="40%">

<img src='./imgs/ROBERTA_LOSS.png' width="40%"> 

<img src='./imgs/XLNET_ACCURACY.png' width="40%">

<img src='./imgs/XLNET_LOSS.png' width="40%"> 

<img src='./imgs/XLNET_LARGE_ACCURACY.png' width="40%">

<img src='./imgs/XLNET_LARGE_LOSS.png' width="40%"> -->

<img src='./imgs/PARAMETERS.png' width="40%">

<img src='./imgs/EVALUATION_LOSS.png' width="40%">

<img src='./imgs/EVALUATION_ACCURACY.png' width="40%">

All models had higher loss values for stsb task. This is possibly because stsb involves understanding context and comparing two different sentences. This possibly means, for stsb the models need for fine-tuning compared to the rest. Among all the models Bert-large-cased had higher losses overall, possibly because it is looking for stricter quality control. This implies the Bert-large-cased needs more fine-tuning compared to the rest.

Overall all models had close to random accuracy for most tasks. This shows without finetuning the models are not reliable. All models had similar accuracy for sst2 dataset. Roberta-base had higher accuracy on mrpc dataset. Whereas all models performed poorly mnli tasks, possibly because it involves reasoning. Surprisingly Roberta-base and XL-net-large-cased performed poorly compared to all the other models at qqp tasks.
- Ipython notebooks can be organized in `notebooks`.

## Ablation study

### Fine-tune v.s. zero-shot

All evaluations in the above sections are zero-shot results — the pretrained models are asked to predict the testing dataset without fine-tuning on it. In this section, we compare the results between fine-tune and zero-shot on selected dataset. We first compared finetuned models and zero-shot models on dataset sst2 and mrpn. The plot below demonsrates that finetuning can improve the model accuracy by almost 100%. 

<p align="left">
    <img src="./imgs/sst_trainedv.s.untrained.png" width="600" height="400">
    <img src="./imgs/mrpc_trainedv.s.untrained.png" width="600" height="400">
</p>

### Half precision v.s. single precision 
One of the most interesting option to run transformer like model is the floating point precision. It is common to use mixed precision (half precision: fp16) to improve training speed. There are mainly 3 reasons to favor mixed precision training in practice:
* It speeds training by almost 2x.  
* It consumes much less GPU memory. In the case of large transformer model such as GPT-J (6B parameters), even single batch size takes 24GB GPU memory in float32 precision. Therefore, it is a must to use mixed precision to fit big transformers in most GPUs unless exploring model parallelism.  
* ZeRO Optimizations which offload GPU memory to CPU during some stages of training only supports half precision. 

Therefore, in this section, we conducted an ablation study on half precision v.s. single precision. The focus is to examine if half precision leads to any accuracy loss. And the following plot demonsrates that there is no significant evaluation accuracy loss when dropping half precision.

<p align="left">
    <img src="./imgs/eval_acc_fp16_fp32.png" width="600" height="300">
</p>

### Base model v.s. large model

One of the gold standard in transformer model is the bigger the better. In this section, we take a closer on this standard on our datasets. We selected BERT-base, BERT-large, XLNet and XLNet-large to conduct the comparison. We also normalized the accuracy by the number of parameters or the per step time to see if there is a diminishing effect to increase model size.

<p align="left">
    <img src="./imgs/base_vs_large.png" width="600" height="300">
    <img src="./imgs/samples_seconds_base_large.png" width="600" height="300">
</p>

The above plots showed that increasing the number of parameters by 3X on BERT model leads to roughly 10% improvement in accuracy. The runtime overhead is about 5X. Therefore, the complexity cost of increasing model parameters is more than the accuracy gain.

### Cased v.s. uncased

Most pretrained models come with two options — cased and uncased. Most uncased models are pretrained on lower-cased English text. In this section, we examine if the choice of cased/uncased pretrained models can lead to a difference in the downstream tasks. 

The plot below shows that there is no significant accuracy difference on tasks sst2 and mrpc between cased BERT and uncased BERT.

<p align="left">
    <img src="./imgs/cased_vs_uncased.png" width="600" height="300">
</p>
