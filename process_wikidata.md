# Procedure of processing wikidata
------
## Overview
| Script        | Description   |
| :---:         | ---           |
| acquire_title_with_blink.py|Use BLINK to get linked wikipedia title of entities. |
| acquire_data_with_linked_titles.py|Acquire QID for each entity using its linked title.   |
| acquire_class_of_entities.py|Get wikidata classes of each entity.    |
|acquire_instances_of_classes.py|Get instances of directly generated classes.|
|process_class_instances_with_flair.py|Use flair model to predict generated instances’ title and remove those that can be correctly predicted.|


## Environment
1. wikidata2df
```
pip install wikidata2df
```
2. BLINK
```
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch -n AttackNER
```


## How to run
Before running each script, change "DATASET_NAME" and "DATASET_TYPE" in it.
Under `AttackNER/process_wikidata` folder.

1. Create folders in test/dev/train set folder.
```
mkdir ./api_results ./entity_wikidata_pair ./none_id_entities ./class_data ./flair_processed_class_instances ./none_instance_classes ./class_instances ./none_class_entities ./titles
```
2. Generate linked titles.
Change "DATASET_TYPE" in acquire_title_with_blink.py.
For test set and dev set, run
```
declare -a start_idx=("0" "3000")
for START_IDX in "${start_idx[@]}"
do
    python acquire_title_with_blink.py ${START_IDX} &
done
```
For train set, run
```
declare -a start_idx=("0" "3000" "6000" "9000" "12000" "15000" "18000" "21000" "24000" "27000" "30000" "33000" "36000" "39000")
for START_IDX in "${start_idx[@]}"
do
    python acquire_title_with_blink.py ${START_IDX} &
done
```
3. Merge all blink_results files into one according to the start_idx.
```
python merge_blink_results.py
```

4. Generate entity QID using its linked title.
```
python acquire_data_with_linked_titles.py
```

5. Generate class data of each entity.
```
python acquire_class_of_entities.py
```

7. Generate instances of each class.
```
python acquire_instances_of_classes.py
```

6. Remove correctly predicted instances with flair.
```
declare -a ent_types=("GPE" "LOC" "ORG" "NORP" "FAC" "EVENT" "WORK_OF_ART" "PRODUCT" "LANGUAGE" "LAW")
for ENT_TYPE in "${ent_types[@]}"
do
    python process_class_instances_with_flair.py ${ENT_TYPE} & 
done 
```

7. Manually delete some unreasonable class_instances data.

8. Generate ontonotes_english_attacked.
under `AttackNER/attack_generation`
Change MODEL in run_cncr_rockner_pipeline.py as "spacy_md", DATASET_TYPE as "train", ARGS_ENTITY_LEVEL_BASIC_CONCURRENT as "dict(attack_num=5, seeds=[10, 11, 12, 13, 14], sampled_pct=[0.2, 0.4, 0.6, 0.8, 1.0])", run:
```bash
CUDA_VISIBLE_DEVICES=1 python run_cncr_rockner_pipeline.py 4
```

Create a folder named ontonotes_english_attacked and copy test.txt, dev.txt and train.txt from ontonotes_english.

under `AttackNER/attack_generation`
```bash
python gen_attacked_training_data.py
```

9. Remove instances used by ontonotes_english_attacked from unseen_flair_processed_class_instances.
under `AttackNER/process_wikidata/ontonotes_english/dev` and `AttackNER/process_wikidata/ontonotes_english/test`
Create a folder named unseen_flair_processed_excl_class_instances

under `AttackNER/process_wikidata`
Change DATASET_TYPE as "test" or "dev".
```bash
python remove_attacked_instances.py
```
Set PROCESSED_CLASS_INSTANCES in run_cncr_rockner_pipeline.py as follows:
```
PROCESSED_CLASS_INSTANCES = "../process_wikidata/" + DATASET_NAME + "/" + DATASET_TYPE \
                            + "/unseen_flair_processed_excl_class_instances/"
```
