import sys
import json
import os

DATASET_NAME = "ontonotes_english"
DATASET_TYPE = "test"
MODEL = "blstm_crf"
ARGS_ENTITY_LEVEL_BASIC = dict(attack_num=5, seeds=[1, 2, 3, 4, 5])
ARGS_ENTITY_LEVEL_BASIC_CONCURRENT = dict(attack_num=5, seeds=[0, 1, 2, 3, 4], sampled_pct=[0.2, 0.4, 0.6, 0.8, 1.0])
ARGS = ARGS_ENTITY_LEVEL_BASIC_CONCURRENT

sys.path.append("../../attack_generation/process_wikidata")
import wikidata_tools
sys.path.append("../../tools")
import adver_pipeline
import adver_tools

MODEL_FILE_FOLDER_NAME = MODEL + "_entity_" + str(ARGS['attack_num']) + "attacks_concurrent_unseen_deleted"
OUTPUT_MAIN_PATH = "../../rockner/" + DATASET_NAME + "/" + DATASET_TYPE + "/" + MODEL_FILE_FOLDER_NAME + "/"
JSONL_FILES_PATH = OUTPUT_MAIN_PATH + "jsonl_files/"
VISUALIZED_RESULTS_PATH = OUTPUT_MAIN_PATH + "visualized_results/"

ENTITY_TYPE_LIST_WITHOUT_PERSON = {"ontonotes_names": ['GPE', 'ORG', 'FAC', 'LOC', 'NORP', 'LAW', 'EVENT',
                                                       'WORK_OF_ART', 'PRODUCT', 'LANGUAGE'],
                                   "conll2003": ["ORG", "LOC"],
                                   "ontonotes_english": ['GPE', 'ORG', 'FAC', 'LOC', 'NORP', 'LAW', 'EVENT',
                                                         'WORK_OF_ART', 'PRODUCT', 'LANGUAGE']
                                   }
STATUS = ["Samespan_Correct", "1_Correct", "2_Correct", "3_Correct", "Samespan_Wrong", "1_Wrong", "2_Wrong",
          "3_Wrong", "None"]

adver_tools.ENTITY_TYPE_LIST = adver_tools.ENTITY_TYPE_LIST_BY_DATASET_NAME[DATASET_NAME]
adver_pipeline.ENTITY_TYPE_LIST_WITHOUT_PERSON = ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]
wikidata_tools.ENTITY_TYPE_LIST_WITHOUT_PERSON = ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]


def load_jsonl_files(path):
    jsonl_data = {k: [] for k in range(ARGS['attack_num'])}
    for attack_round in range(ARGS['attack_num']):
        with open(path + str(attack_round) + '.jsonl', 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                jsonl_data[attack_round].append(dict(original=data['original'],
                                                     adver=data['adver'],
                                                     entities_with_derivation=data['entities_with_derivation'],
                                                     predictions=data['predictions']))
    return jsonl_data


def make_visualized_sent(ori_sent_str, entities):
    sent_str = ""
    sent = ori_sent_str.split()
    processed_ents = sorted([ent for ent in entities if ent[3] in adver_tools.ENTITY_TYPE_LIST], key=lambda x: x[1])
    if not processed_ents:
        return ori_sent_str
    start_idx = 0
    for ent_idx, ent in enumerate(processed_ents):
        sent_str += " ".join(sent[start_idx:ent[1]]) + " < " + " ".join(sent[ent[1]: ent[2]+1]) + " | " + ent[3] + " > "
        start_idx = ent[2] + 1
        if ent_idx == len(processed_ents) - 1:
            sent_str += " ".join(sent[ent[2]+1:])
    return sent_str




def visualize_data(path, jsonl_data):
    for attack_round in range(ARGS['attack_num']):
        with open(path + str(attack_round) + '.txt', 'w+') as f:
            for data in jsonl_data[attack_round]:
                if not data['adver']['entities']:
                    continue
                original = data['original']
                original_sent_str = original['sentence']
                original_sampled_ents = original['sampled_entities']
                vis_ori_sent_str = make_visualized_sent(original_sent_str, original_sampled_ents)
                sent_id = original['all_entities'][0][0]
                adver = data['adver']
                adver_sent_str = adver['sentence']
                adver_entities = adver['entities']
                vis_adver_sent_str = make_visualized_sent(adver_sent_str, adver_entities)
                predictions = data['predictions']
                processed_preds = [[sent_id, pred[0], pred[1], pred[2]] for pred in predictions]
                vis_pred_sent_str = make_visualized_sent(adver_sent_str, processed_preds)
                entities_with_derivation = data['entities_with_derivation']
                # attack_succeed_flag = data['attack_succeed']
                f.write("\n------------------------ SENT_ID: " + str(sent_id) + " ------------------------\n")
                f.write(vis_ori_sent_str + '\n' + vis_adver_sent_str + '\n' + vis_pred_sent_str + '\n\n')
                for ent_w_deriv, adver_ent in zip(entities_with_derivation, adver_entities):
                    f.write(ent_w_deriv['ent_text'] + " \n")
                    f.write("\tlinked_title: " + ent_w_deriv['linked_title'] + " | " + ent_w_deriv['ent_id'] + "\n")
                    f.write("\tinstance: " + list(ent_w_deriv['instance'].values())[0] + " | " + list(ent_w_deriv['instance'].keys())[0] + "\n")
                    if ent_w_deriv['derive_from'] != 'N/A':
                        f.write("\tclass: " + ent_w_deriv['derive_from']['class_title'] + " | " + ent_w_deriv['derive_from']['class_id'] + "\n")
                    else:
                        f.write("\tclass: N/A\n")
                    if [adver_ent[1], adver_ent[2], adver_ent[3]] in predictions:
                        f.write("\tattack_succeed: Fasle" + "\n")
                    else:
                        f.write("\tattack_succeed: True" + "\n")


def main():
    jsonl_data = load_jsonl_files(JSONL_FILES_PATH)

    if not os.path.exists(VISUALIZED_RESULTS_PATH):
        os.system("mkdir -p " + VISUALIZED_RESULTS_PATH)

    visualize_data(VISUALIZED_RESULTS_PATH, jsonl_data)


if __name__ == '__main__':
    main()

