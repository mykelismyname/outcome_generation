**Identify exisiting outcomes based on the inherent labels and count how often each identified outcome appears in the data**

python prepare_data.py --outcome_frequency --data data/ebm-comet/

**Train the model to learn fill in missing outcomes in prompt using a pretrained mlm**

python train_mlm.py --data data/ebm-comet/ --output_dir **[specify output dir]** --pretrained_model **[specify a pretrained mlm e.g. allenai/scibert_scivocab_uncased]** --per_device_train_batch_size 16 --per_device_test_batch_size 16 --save_steps 1000 --do_train --do_eval --do_fill --custom_mask

**TASK the model to fill in missing outcomes using a pretrained mlm**

python train_mlm.py --data data/ebm-comet/train.txt --output_dir **[specify output dir]** --pretrained_model output/roberta_custom_mlm_16_20/ --fill_evaluation --recall_metric **['exact_match' or 'partial_match']** --mention_frequency **[Path to a json file in which there is the frequency of occurrence for each outcome in the data]**

**Note:-**
custom mask: is passed in order to specificlly only mask the target entities or tokens in the data and for this case, outcomes are the target entities.

metric: partial_match -  Given 4 outcomes of span length 3, if model doesn't recall all 3 tokens for each span e.g. (1/3 for outcome 1, 2/3 for outcome 2
        1/3 for outcome 4, and 3/3 for outcome 4. accuracy will be determined by an average accuracy computed as (1/3 + 2/3 + 1/3 + 3/3)/4 = 1/2
        
metric: exact match - For the same example above, exact match accuracy would be 1/4, because only 1 outcome was fully recalled
_
