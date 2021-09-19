import sys
import time
import os
from tqdm import tqdm

DATASET_NAME = "ontonotes_english"
DATASET_TYPE = "dev"
MODEL = "roberta_crf_aug_mixirng_up"
MODEL_PATH = "../model_training/resources/taggers/" + MODEL
ARGS_ENTITY_LEVEL_BASIC = dict(attack_num=5, seeds=[1, 2, 3, 4, 5])
ARGS = ARGS_ENTITY_LEVEL_BASIC
USE_FILTER = True
REPLACE_MARK = "context"  # both or context

sys.path.append("../../attack_generation/process_wikidata")
import wikidata_tools
sys.path.append("../../tools")
import adver_pipeline
import adver_tools

DATA_FILE_PATH = "../../data/" + DATASET_NAME + "/" + DATASET_TYPE + ".txt"
MODEL_FILE_FOLDER_NAME = MODEL + "_entity_" + str(ARGS['attack_num']) + "attacks_concurrent_unseen_deleted"
OUTPUT_MAIN_PATH = "../../rockner/" + DATASET_NAME + "/" + DATASET_TYPE + "/" + MODEL_FILE_FOLDER_NAME + "/"
CONTEXT_REPLACED_ROBUSTNESS_PATH = OUTPUT_MAIN_PATH + "context_replaced_robustness/"

ONTOROCK_E_PATH = "../../OntoRock/" + DATASET_TYPE + "/OntoRock-E/"

if REPLACE_MARK == "context":
    OUTPUT_PATH = "../../OntoRock/" + DATASET_TYPE + "/OntoRock-C/"
else:
    OUTPUT_PATH = "../../OntoRock/" + DATASET_TYPE + "/OntoRock-F/"

if USE_FILTER:
    CONTEXT_REPLACED_DATA_PATH = OUTPUT_PATH + "filtered.txt"
    ROBUSTNESS_PATH = OUTPUT_MAIN_PATH + "context_replaced_robustness/" \
                      + REPLACE_MARK + "_replaced_filtered"
else:
    CONTEXT_REPLACED_DATA_PATH = OUTPUT_PATH + "unfiltered.txt"
    ROBUSTNESS_PATH = OUTPUT_MAIN_PATH + "context_replaced_robustness/" \
                      + REPLACE_MARK + "_replaced"

ENTITY_TYPE_LIST_WITHOUT_PERSON = {"ontonotes_names": ['GPE', 'ORG', 'FAC', 'LOC', 'NORP', 'LAW', 'EVENT',
                                                       'WORK_OF_ART', 'PRODUCT', 'LANGUAGE'],
                                   "conll2003": ["ORG", "LOC"],
                                   "ontonotes_english": ['GPE', 'ORG', 'FAC', 'LOC', 'NORP', 'LAW', 'EVENT',
                                                         'WORK_OF_ART', 'PRODUCT', 'LANGUAGE']
                                   }

adver_tools.ENTITY_TYPE_LIST = adver_tools.ENTITY_TYPE_LIST_BY_DATASET_NAME[DATASET_NAME]
adver_pipeline.ENTITY_TYPE_LIST_WITHOUT_PERSON = ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]
wikidata_tools.ENTITY_TYPE_LIST_WITHOUT_PERSON = ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]


def output_results(path, all_sents, all_gt_tags, all_pred_tags):
    with open(path + '.results', 'w+') as f:
        for sent, gt_tags, pred_tags in zip(all_sents, all_gt_tags, all_pred_tags):
            for token, tag, pred_tag in zip(sent, gt_tags, pred_tags):
                # token     correct_tag     predicted_tag
                f.write(token + '\t' + tag + '\t' + pred_tag + '\n')
            f.write('\n')


def main():
    # attack_round = 4 # only use file with 100% replaced entities
    adver_pipeline.model_init(MODEL, model_path=MODEL_PATH, device="cuda:0")
    all_sents, all_tags = adver_tools.read_data(CONTEXT_REPLACED_DATA_PATH)
    entities_by_type, entities_by_sid, entities_set = adver_tools.update_NER_dict(all_sents, all_tags)
    all_gt_tags = []
    all_predicted_tags = []
    all_predictions = []
    for sent_id, sent in enumerate(tqdm(all_sents)):
        gt_tags = all_tags[sent_id]
        if len(entities_by_sid[sent_id]) == 0:
            pred_tags = all_tags[sent_id]
            all_predictions.append(None)
        else:
            predictions = adver_pipeline.predict_sentence(sent, MODEL)
            pred_tags = adver_tools.get_all_tags(sent, predictions)
            all_predictions.append(predictions)
        all_gt_tags.append(gt_tags)
        all_predicted_tags.append(pred_tags)
    if not os.path.exists(CONTEXT_REPLACED_ROBUSTNESS_PATH):
        os.system("mkdir -p " + CONTEXT_REPLACED_ROBUSTNESS_PATH)

    output_results(ROBUSTNESS_PATH, all_sents, all_gt_tags, all_predicted_tags)

    time.sleep(5)

    os.system("python conlleval.py <" + ROBUSTNESS_PATH + ".results > "
              + ROBUSTNESS_PATH + ".f1_report.txt")


if __name__ == '__main__':
    main()
