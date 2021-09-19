import sys
import os
from tqdm import tqdm

DATASET_NAME = "ontonotes_english"
DATASET_TYPE = "dev"
MODEL = "roberta_crf"
MODEL_PATH = "../model_training/resources/taggers/" + MODEL
ARGS_ENTITY_LEVEL_BASIC = dict(attack_num=5, seeds=[1, 2, 3, 4, 5])
ARGS = ARGS_ENTITY_LEVEL_BASIC

sys.path.append("../../attack_generation/process_wikidata")
import wikidata_tools
sys.path.append("../../tools")
import adver_pipeline
import adver_tools

DATA_FILE_PATH = "../../data/" + DATASET_NAME + "/" + DATASET_TYPE + ".txt"
MODEL_FILE_FOLDER_NAME = MODEL + "_entity_" + str(ARGS['attack_num']) + "attacks_concurrent_unseen_deleted"
OUTPUT_MAIN_PATH = "../../rockner/" + DATASET_NAME + "/" + DATASET_TYPE + "/" + MODEL_FILE_FOLDER_NAME + "/"

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
    with open(path + DATASET_TYPE + '.results', 'w+') as f:
        for sent, gt_tags, pred_tags in zip(all_sents, all_gt_tags, all_pred_tags):
            for token, tag, pred_tag in zip(sent, gt_tags, pred_tags):
                # token     correct_tag     predicted_tag
                f.write(token + '\t' + tag + '\t' + pred_tag + '\n')
            f.write('\n')


def main():
    if not os.path.exists(OUTPUT_MAIN_PATH):
        os.system("mkdir " + OUTPUT_MAIN_PATH)

    adver_pipeline.model_init(MODEL, model_path=MODEL_PATH, device="cuda:0")
    all_sents, all_tags = adver_tools.read_data(DATA_FILE_PATH)
    entities_by_type, entities_by_sid, entities_set = adver_tools.update_NER_dict(all_sents, all_tags)
    all_gt_tags = []
    all_predicted_tags = []
    all_predictions = []
    for sent_id, sent in tqdm(enumerate(all_sents), total=len(all_sents)):
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
    output_results(OUTPUT_MAIN_PATH, all_sents, all_gt_tags, all_predicted_tags)


if __name__ == '__main__':
    main()
