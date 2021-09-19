# Model Training and Evaluation


## Model Training 
In this part, we train several models on original ontonotes_english training set and augmented ontonotes_english training set.

### Falir 

under project folder `cd AttackNER/model_training`
```bash 
CUDA_VISIBLE_DEVICES=1 python train_flair.py
```
self_trained_flair_cpu: mini_batch_size=32, train_with_dev=True, embeddings_storage_mode="none"
self_trained_flair_gpu: mini_batch_size=64, train_with_dev=False, embeddings_storage_mode="none"

### Flair (+data aug)

under project folder `cd AttackNER/model_training`
Change data_folder in train_flair.py as "../data/ontonotes_english_attacked" and model_name as "self_trained_flair_gpu_attacked".
```bash 
CUDA_VISIBLE_DEVICES=1 python train_flair.py
```
self_trained_flair_gpu: mini_batch_size=64, train_with_dev=False, embeddings_storage_mode="none"

### Pytorch-lstmcrf

under project folder `cd pytorch_lstmcrf` 
For blstm_crf model, run:
```bash
python trainer.py --device=cuda:1 --dataset=ontonotes_english --model_folder=blstm_crf --batch_size=20 --max_no_incre=10
```

For bert_crf model, run:
```bash
python transformers_trainer.py --device=cuda:2 --dataset=ontonotes_english --model_folder=../AttackNER/model_training/resources/taggers/bert_crf/ --embedder_type=bert-base-cased --batch_size=20 --max_no_incre=5
```
[test set Total] Prec.: 90.98, Rec.: 89.94, F1: 90.46

For roberta_crf model, run:
```bash
python transformers_trainer.py --device=cuda:2 --dataset=ontonotes_english --model_folder=../AttackNER/model_training/resources/taggers/roberta_crf/ --embedder_type=roberta-base --batch_size=20 --max_no_incre=5
```
[test set Total] Prec.: 92.30, Rec.: 91.99, F1: 92.15

### Pytorch-lstmcrf (+data aug)

From `project/AttackNER/data` copy augmented data foler into `project/pytorch_lstmcrf/data`.

under project folder `cd pytorch_lstmcrf`, 
For blstm_crf_attacked model, run:
```bash
python trainer.py --device=cuda:0 --dataset=ontonotes_english_attacked --model_folder=blstm_crf_attacked --batch_size=20 --max_no_incre=10
```
[test set Total] Prec.: 83.47, Rec.: 82.49, F1: 82.98

For bert_crf_attacked model, run:
```bash
CUDA_VISIBLE_DEVICES=3 python transformers_trainer.py --device=cuda:0 --dataset=ontonotes_english_attacked --model_folder=bert_crf_attacked --embedder_type=bert-base-cased --batch_size=20 --max_no_incre=5
```
[test set Total] Prec.: 90.14, Rec.: 90.85, F1: 90.50

For roberta_crf_attacked model, run:
```bash
python transformers_trainer.py --device=cuda:2 --dataset=ontonotes_english_attacked --model_folder=roberta_crf_attacked --embedder_type=roberta-base --batch_size=20 --max_no_incre=5
```
[test set Total] Prec.: 91.57, Rec.: 90.71, F1: 91.14

For roberta_crf_aug_random_chars model, run:
```bash
python transformers_trainer.py --device=cuda:1 --dataset=ontonotes_english_aug_random_chars --model_folder=roberta_crf_aug_random_chars --embedder_type=roberta-base --batch_size=10 --max_no_incre=5
```
[test set Total] Prec.: 90.47, Rec.: 91.56, F1: 91.01

For roberta_crf_aug_entity_switching model, run:
```bash
python transformers_trainer.py --device=cuda:1 --dataset=ontonotes_english_aug_entity_switching --model_folder=roberta_crf_aug_entity_switching --embedder_type=roberta-base --batch_size=20 --max_no_incre=5
```
[test set Total] Prec.: 91.52, Rec.: 90.92, F1: 91.22

For roberta_crf_aug_mix_up model, run:
```bash
CUDA_VISIBLE_DEVICES=2 python transformers_trainer.py --device=cuda:0 --dataset=ontonotes_english_aug_mix_up --model_folder=roberta_crf_aug_mix_up --embedder_type=roberta-base --batch_size=10 --max_no_incre=5
```
[test set Total] Prec.: 91.37, Rec.: 91.27, F1: 91.32

For roberta_crf_aug_random_masking model, run:
```bash
CUDA_VISIBLE_DEVICES=3 python transformers_trainer.py --device=cuda:0 --dataset=ontonotes_english_aug_random_masking --model_folder=roberta_crf_aug_random_masking --embedder_type=roberta-base --batch_size=20 --max_no_incre=5
```
[test set Total] Prec.: 91.95, Rec.: 92.75, F1: 92.35

For roberta_crf_aug_mixing_up model, run:
```bash
python transformers_trainer.py --device=cuda:1 --dataset=ontonotes_english_aug_mixing_up --model_folder=roberta_crf_aug_mixing_up --embedder_type=roberta-base --batch_size=20 --max_no_incre=5
```
[test set Total] Prec.: 91.48, Rec.: 91.78, F1: 91.63

## Model Evaluation
We evaluate above models on test and dev sets with replacing 20%, 40%, 60%, 80%, 100% entities.

### Seeds:
flair_entity_5attacks: [1, 2, 3, 4, 5]
the unattacked models use: [0, 1, 2, 3, 4]
the attacked models use: [5, 6, 7, 8, 9]
Use seed=14 to generate attacked training data.

### Flair

1. Create environment and install flair.
```bash
conda create -n rockner_flair python=3.7
```
```bash
conda activate rockner_flair

pip install flair
pip isntall numpy
pip install transformers==3.3.1
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install happytransformer==1.1.3
```

under project folder `cd AttackNER/attack_generation`
2. Evaluate raw dev and test with models.
Change MODEL in run_raw_dataset_eval.py as "self_trained_flair", DATASET_TYPE as "dev" or "test".
```bash
python run_raw_dataset_eval.py
```

Generate F1 report for dev set.
```bash
python conlleval.py <./rockner/ontonotes_english/dev/self_trained_flair_entity_5attacks_concurrent_unseen_deleted/dev.results >./rockner/ontonotes_english/dev/self_trained_flair_entity_5attacks_concurrent_unseen_deleted/dev.f1_report.txt
```

Generate F1 report for test set.
```bash
python conlleval.py <./rockner/ontonotes_english/test/self_trained_flair_entity_5attacks_concurrent_unseen_deleted/test.results >./rockner/ontonotes_english/test/self_trained_flair_entity_5attacks_concurrent_unseen_deleted/test.f1_report.txt
```

3. Evaluate rockner test and rockner dev with models.
Change MODEL in run_cncr_rockner_pipeline.py as "self_trained_flair", DATASET_TYPE as "dev" or "test", run:
```bash
declare -a attack_round=("0" "1" "2" "3" "4")
for ATTACK_ROUND in "${attack_round[@]}"
do
    CUDA_VISIBLE_DEVICES=0 python run_cncr_rockner_pipeline.py ${ATTACK_ROUND} & 
done 
```

4. Generate visualized results.
Change MODEL in visualize_rockner.py as "self_trained_flair", DATASET_TYPE as "dev" or "test", run:
```bash
python visualize_rockner.py
```

5. Generate F1 reports.
Change MODEL in run_conlleval.py as "self_trained_flair", DATASET_TYPE as "dev" or "test", run:
```bash
python run_conlleval.py
```

### Spacy

1. Create a new environment named "rockner_spacy" and install spacy.
```bash
conda create -n rockner_spacy python=3.7
```
```bash
conda activate rockner_spacy
conda install -c conda-forge spacy
pip install transformers==3.3.1
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install happytransformer==1.1.3
pip install -U spacy[cuda101]
```

2. Download models.
```bash
python -m spacy download en_core_web_lg
```
```bash
python -m spacy download en_core_web_md
```

Following commands are executed under folder `AttackNER/attack_generation`

3. Evaluate raw dev and test with models.
Change MODEL in run_raw_dataset_eval.py as "spacy_lg" or "spacy_md", DATASET_TYPE as "dev" or "test", run:
```bash
python run_raw_dataset_eval.py
```
Generate F1 report for dev set.
```bash
python conlleval.py <./rockner/ontonotes_english/dev/spacy_lg_entity_5attacks_concurrent_unseen_deleted/dev.results >./rockner/ontonotes_english/dev/spacy_lg_entity_5attacks_concurrent_unseen_deleted/dev.f1_report.txt
```
Generate F1 report for test set.
```bash
python conlleval.py <./rockner/ontonotes_english/test/spacy_lg_entity_5attacks_concurrent_unseen_deleted/test.results >./rockner/ontonotes_english/test/spacy_lg_entity_5attacks_concurrent_unseen_deleted/test.f1_report.txt
```

4. Evaluate rockner test and rockner dev with models.
Change MODEL in run_cncr_rockner_pipeline.py as "spacy_lg" or "spacy_md", DATASET_TYPE as "dev" or "test", run:
```bash
declare -a attack_round=("0" "1" "2" "3" "4")
for ATTACK_ROUND in "${attack_round[@]}"
do
    CUDA_VISIBLE_DEVICES=3 python run_cncr_rockner_pipeline.py ${ATTACK_ROUND} & 
done 
```

5. Generate visualized results.
Change MODEL in visualize_rockner.py as "spacy_lg" or "spacy_md", DATASET_TYPE as "dev" or "test", run:
```bash
python visualize_rockner.py
```

6. Generate F1 reports.
Change MODEL in run_conlleval.py as "spacy_lg" or "spacy_md", DATASET_TYPE as "dev" or "test", run:
```bash
python run_conlleval.py
```

### Stanza

1. Create a new environment named "rockner_stanza" and install stanza.
```bash
conda create -n rockner_stanza python=3.7
```
```bash
conda activate rockner_stanza
pip install stanza
pip install transformers==3.3.1
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install happytransformer==1.1.3
```

Following commands are executed under folder `AttackNER/attack_generation`

2. Evaluate raw dev and test with models.
Change MODEL in run_raw_dataset_eval.py as "stanza" , DATASET_TYPE as "dev" or "test", run:
```bash
python run_raw_dataset_eval.py
```
Generate F1 report for dev set.
```bash
python conlleval.py <./rockner/ontonotes_english/dev/stanza_entity_5attacks_concurrent_unseen_deleted/dev.results >./rockner/ontonotes_english/dev/stanza_entity_5attacks_concurrent_unseen_deleted/dev.f1_report.txt
```
Generate F1 report for test set.
```bash
python conlleval.py <./rockner/ontonotes_english/test/stanza_entity_5attacks_concurrent_unseen_deleted/test.results >./rockner/ontonotes_english/test/stanza_entity_5attacks_concurrent_unseen_deleted/test.f1_report.txt
```

3. Evaluate rockner test and rockner dev with model.
Change MODEL in run_cncr_rockner_pipeline.py as "stanza", DATASET_TYPE as "dev" or "test", run:
```bash
declare -a attack_round=("0" "1" "2" "3" "4")
for ATTACK_ROUND in "${attack_round[@]}"
do
    CUDA_VISIBLE_DEVICES=1 python run_cncr_rockner_pipeline.py ${ATTACK_ROUND} & 
done 
```

4. Generate visualized results.
Change MODEL in visualize_rockner.py as "stanza", DATASET_TYPE as "dev" or "test", run:
```bash
python visualize_rockner.py
```

5. Generate F1 reports.
Change MODEL in run_conlleval.py as "stanza", DATASET_TYPE as "dev" or "test", run:
```bash
python run_conlleval.py
```

### Pytorch-lstmcrf
There are three models: blstm_crf, bert_crf, roberta_crf.

1. Create a new environment named "rockner_crf" and install compatible pytorch and tranformers.
```bash
conda create -n rockner_crf python=3.7
```
```bash
conda activate rockner_crf
pip install transformers==3.3.1
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install happytransformer==1.1.3
```

2. Evaluate raw dev and test with models.
Change MODEL in run_raw_dataset_eval.py as "bert_crf" or "roberta_crf" , DATASET_TYPE as "dev" or "test", run:
```bash
CUDA_VISIBLE_DEVICES=3 python run_raw_dataset_eval.py 
```
Generate F1 report for dev set.
```bash
python conlleval.py <./rockner/ontonotes_english/dev/bert_crf_entity_5attacks_concurrent_unseen_deleted/dev.results >./rockner/ontonotes_english/dev/bert_crf_entity_5attacks_concurrent_unseen_deleted/dev.f1_report.txt
```
```bash
python conlleval.py <./rockner/ontonotes_english/dev/roberta_crf_entity_5attacks_concurrent_unseen_deleted/dev.results >./rockner/ontonotes_english/dev/roberta_crf_entity_5attacks_concurrent_unseen_deleted/dev.f1_report.txt
```
```bash
python conlleval.py <./rockner/ontonotes_english/dev/blstm_crf_entity_5attacks_concurrent_unseen_deleted/dev.results >./rockner/ontonotes_english/dev/blstm_crf_entity_5attacks_concurrent_unseen_deleted/dev.f1_report.txt
```
Generate F1 report for test set.
```bash
python conlleval.py <./rockner/ontonotes_english/test/bert_crf_entity_5attacks_concurrent_unseen_deleted/test.results >./rockner/ontonotes_english/test/bert_crf_entity_5attacks_concurrent_unseen_deleted/test.f1_report.txt
```
```bash
python conlleval.py <./rockner/ontonotes_english/test/roberta_crf_entity_5attacks_concurrent_unseen_deleted/test.results >./rockner/ontonotes_english/test/roberta_crf_entity_5attacks_concurrent_unseen_deleted/test.f1_report.txt
```
```bash
python conlleval.py <./rockner/ontonotes_english/test/blstm_crf_entity_5attacks_concurrent_unseen_deleted_excl/test.results >./rockner/ontonotes_english/test/blstm_crf_entity_5attacks_concurrent_unseen_deleted_excl/test.f1_report.txt
```

3. Evaluate rockner test and rockner dev with model.
Change MODEL in run_cncr_rockner_pipeline.py as "bert_crf" or "roberta_crf", DATASET_TYPE as "dev" or "test", run:
```bash
declare -a attack_round=("0" "1" "2" "3" "4")
for ATTACK_ROUND in "${attack_round[@]}"
do
    CUDA_VISIBLE_DEVICES=1 python run_cncr_rockner_pipeline.py ${ATTACK_ROUND} & 
done 
```

4. Generate visualized results.
Change MODEL in visualize_rockner.py as "bert_crf" or "roberta_crf", DATASET_TYPE as "dev" or "test", run:
```bash
python visualize_rockner.py
```

5. Generate F1 reports.
Change MODEL in run_conlleval.py as "bert_crf" or "roberta_crf", DATASET_TYPE as "dev" or "test", run:
```bash
python run_conlleval.py
```

### Flair (+data aug)
```bash
conda activate rockner_flair
```
1. Evaluate raw dev and test with models.
Change MODEL in run_raw_dataset_eval.py as "self_trained_flair_attacked", DATASET_TYPE as "dev" or "test".
```bash
python run_raw_dataset_eval.py
```

Generate F1 report for dev set.
```bash
python conlleval.py <./rockner/ontonotes_english/dev/self_trained_flair_attacked_entity_5attacks_concurrent_unseen_deleted/dev.results >./rockner/ontonotes_english/dev/self_trained_flair_attacked_entity_5attacks_concurrent_unseen_deleted/dev.f1_report.txt
```

Generate F1 report for test set.
```bash
python conlleval.py <./rockner/ontonotes_english/test/self_trained_flair_attacked_entity_5attacks_concurrent_unseen_deleted/test.results >./rockner/ontonotes_english/test/self_trained_flair_attacked_entity_5attacks_concurrent_unseen_deleted/test.f1_report.txt
```

2. Evaluate rockner test and rockner dev with models.
Change MODEL in run_cncr_rockner_pipeline.py as "self_trained_flair_attacked", DATASET_TYPE as "dev" or "test", run:
```bash
declare -a attack_round=("0" "1" "2" "3" "4")
for ATTACK_ROUND in "${attack_round[@]}"
do
    CUDA_VISIBLE_DEVICES=2 python run_cncr_rockner_pipeline.py ${ATTACK_ROUND} & 
done 
```

3. Generate visualized results.
Change MODEL in visualize_rockner.py as "self_trained_flair_attacked", DATASET_TYPE as "dev" or "test", run:
```bash
python visualize_rockner.py
```

4. Generate F1 reports.
Change MODEL in run_conlleval.py as "self_trained_flair_attacked", DATASET_TYPE as "dev" or "test", run:
```bash
python run_conlleval.py
```


### Pytorch-lstmcrf (+data aug)

For "bert_crf_attacked", "roberta_crf_attacked", "roberta_crf_aug_entity_switching" and "roberta_crf_aug_random_chars": 
```bash
conda activate rockner_crf
```

For "blstm_crf_attacked":
```bash
conda activate rockner_blstm
```

1. Evaluate raw dev and test with models.
Change MODEL in run_raw_dataset_eval.py as "bert_crf_attacked", "roberta_crf_attacked", "blstm_crf_attacked", "roberta_crf_aug_entity_switching" or "roberta_crf_aug_random_chars", DATASET_TYPE as "dev" or "test", run:
```bash
CUDA_VISIBLE_DEVICES=3 python run_raw_dataset_eval.py 
```
Generate F1 report for dev set.
```bash
python conlleval.py <./rockner/ontonotes_english/dev/bert_crf_attacked_entity_5attacks_concurrent_unseen_deleted_excl/dev.results >./rockner/ontonotes_english/dev/bert_crf_attacked_entity_5attacks_concurrent_unseen_deleted_excl/dev.f1_report.txt
```
```bash
python conlleval.py <./rockner/ontonotes_english/dev/roberta_crf_attacked_entity_5attacks_concurrent_unseen_deleted_excl/dev.results >./rockner/ontonotes_english/dev/roberta_crf_attacked_entity_5attacks_concurrent_unseen_deleted_excl/dev.f1_report.txt
```
```bash
python conlleval.py <./rockner/ontonotes_english/dev/blstm_crf_attacked_entity_5attacks_concurrent_unseen_deleted_excl/dev.results >./rockner/ontonotes_english/dev/blstm_crf_attacked_entity_5attacks_concurrent_unseen_deleted_excl/dev.f1_report.txt
```
```bash
python conlleval.py <./rockner/ontonotes_english/dev/roberta_crf_aug_entity_switching_entity_5attacks_concurrent_unseen_deleted/dev.results >./rockner/ontonotes_english/dev/roberta_crf_aug_entity_switching_entity_5attacks_concurrent_unseen_deleted/dev.f1_report.txt
```
```bash
python conlleval.py <./rockner/ontonotes_english/dev/roberta_crf_aug_random_chars_entity_5attacks_concurrent_unseen_deleted/dev.results >./rockner/ontonotes_english/dev/roberta_crf_aug_random_chars_entity_5attacks_concurrent_unseen_deleted/dev.f1_report.txt
```
```bash
python conlleval.py <./rockner/ontonotes_english/dev/roberta_crf_aug_mix_up_entity_5attacks_concurrent_unseen_deleted/dev.results >./rockner/ontonotes_english/dev/roberta_crf_aug_mix_up_entity_5attacks_concurrent_unseen_deleted/dev.f1_report.txt
```
```bash
python conlleval.py <./rockner/ontonotes_english/dev/roberta_crf_aug_random_masking_entity_5attacks_concurrent_unseen_deleted/dev.results >./rockner/ontonotes_english/dev/roberta_crf_aug_random_masking_entity_5attacks_concurrent_unseen_deleted/dev.f1_report.txt
```
```bash
python conlleval.py <./rockner/ontonotes_english/dev/roberta_crf_aug_mixing_up_entity_5attacks_concurrent_unseen_deleted/dev.results >./rockner/ontonotes_english/dev/roberta_crf_aug_mixing_up_entity_5attacks_concurrent_unseen_deleted/dev.f1_report.txt
```

Generate F1 report for test set.
```bash
python conlleval.py <./rockner/ontonotes_english/test/bert_crf_attacked_entity_5attacks_concurrent_unseen_deleted/test.results >./rockner/ontonotes_english/test/bert_crf_attacked_entity_5attacks_concurrent_unseen_deleted/test.f1_report.txt
```
```bash
python conlleval.py <./rockner/ontonotes_english/test/roberta_crf_attacked_entity_5attacks_concurrent_unseen_deleted_excl/test.results >./rockner/ontonotes_english/test/roberta_crf_attacked_entity_5attacks_concurrent_unseen_deleted_excl/test.f1_report.txt
```
```bash
python conlleval.py <./rockner/ontonotes_english/test/blstm_crf_attacked_entity_5attacks_concurrent_unseen_deleted_excl/test.results >./rockner/ontonotes_english/test/blstm_crf_attacked_entity_5attacks_concurrent_unseen_deleted_excl/test.f1_report.txt
```
```bash
python conlleval.py <./rockner/ontonotes_english/test/roberta_crf_aug_entity_switching_entity_5attacks_concurrent_unseen_deleted_excl/test.results >./rockner/ontonotes_english/test/roberta_crf_aug_entity_switching_entity_5attacks_concurrent_unseen_deleted_excl/test.f1_report.txt
```
```bash
python conlleval.py <./rockner/ontonotes_english/test/roberta_crf_aug_random_chars_entity_5attacks_concurrent_unseen_deleted_excl/test.results >./rockner/ontonotes_english/test/roberta_crf_aug_random_chars_entity_5attacks_concurrent_unseen_deleted_excl/test.f1_report.txt
```
```bash
python conlleval.py <./rockner/ontonotes_english/test/roberta_crf_aug_mix_up_entity_5attacks_concurrent_unseen_deleted_excl/test.results >./rockner/ontonotes_english/test/roberta_crf_aug_mix_up_entity_5attacks_concurrent_unseen_deleted_excl/test.f1_report.txt
```
```bash
python conlleval.py <./rockner/ontonotes_english/test/roberta_crf_aug_random_masking_entity_5attacks_concurrent_unseen_deleted/test.results >./rockner/ontonotes_english/test/roberta_crf_aug_random_masking_entity_5attacks_concurrent_unseen_deleted/test.f1_report.txt
```
```bash
python conlleval.py <./rockner/ontonotes_english/test/roberta_crf_aug_mixing_up_entity_5attacks_concurrent_unseen_deleted/test.results >./rockner/ontonotes_english/test/roberta_crf_aug_mixing_up_entity_5attacks_concurrent_unseen_deleted/test.f1_report.txt
```

2. Evaluate rockner test and rockner dev with model.
Change MODEL in run_cncr_rockner_pipeline.py as "bert_crf_attacked" or "roberta_crf_attacked", DATASET_TYPE as "dev" or "test", run:
```bash
declare -a attack_round=("0" "1" "2" "3" "4")
for ATTACK_ROUND in "${attack_round[@]}"
do
    CUDA_VISIBLE_DEVICES=2 python run_cncr_rockner_pipeline.py ${ATTACK_ROUND} & 
done 
```

3. Generate visualized results.
Change MODEL in run_raw_dataset_eval.py as "bert_crf_attacked", "roberta_crf_attacked", "blstm_crf_attacked", "roberta_crf_aug_entity_switching" or "roberta_crf_aug_random_chars", DATASET_TYPE as "dev" or "test", run:
```bash
python visualize_rockner.py
```

4. Generate F1 reports.
Change MODEL in run_raw_dataset_eval.py as "bert_crf_attacked", "roberta_crf_attacked", "blstm_crf_attacked", "roberta_crf_aug_entity_switching" or "roberta_crf_aug_random_chars", DATASET_TYPE as "dev" or "test", run:
```bash
python run_conlleval.py
```


## Evaluation on Context Replaced Data
In this part, we generate data with replacing contextual verb, noun, adverb and adjective and filtered sentences with roBERTa model, then evaluation models on generated data.

### Data Generation
under folder `project/AttackNER/attack_generation`
Change `DATASET_TYPE` in gen_context_replaced_data_with_bert.py as "test" or "dev", `USE_FILTER` as False or True
`REPLACE_MARK` as "both", "entity" or "context", run:
```bash
CUDA_VISIBLE_DEVICES=1 python gen_context_replaced_data_with_bert.py
```
| USE_FILTER | REPLACE_MARK | Generated_file | Description |
| :---: | --- | --- | --- |
| True | both | dev.both_replaced_filterd.txt & test.both_replaced_filterd.txt | Generate data by replacing entities and context and filtering with BLSTM. |
| False | both | dev.both_replaced.txt & test.both_replaced.txt | Generate data by replacing entities and context. |
| True | context | dev.context_replaced_filtered.txt & test.context_replaced_filtered.txt | Generate data by replacing context and filtering with BLSTM. |
| False | context | dev.context_replaced.txt & test.context_replaced.txt | Generate data by replacing context. |

### Model Evaluation
under folder `project/AttackNER/attack_generation`
Change `DATASET_TYPE` in eval_context_replaced_data.py as "test" or "dev", `USE_FILTER` as False or True
`REPLACE_MARK` as "both", "entity" or "context", run:
```bash
CUDA_VISIBLE_DEVICES=1 python eval_context_replaced_data.py
```
