import random
import sys
import os
import json
import time
import pandas as pd
from tqdm import tqdm
from copy import deepcopy

DATASET_NAME = "ontonotes_english"
DATASET_TYPE = "test"
MODEL = "stanza"
MODEL_PATH = "../model_training/resources/taggers/" + MODEL
ARGS_ENTITY_LEVEL_BASIC = dict(attack_num=5, seeds=[1, 2, 3, 4, 5])
ARGS_ENTITY_LEVEL_BASIC_CONCURRENT = dict(attack_num=5, seeds=[0, 1, 2, 3, 4], sampled_pct=[0.2, 0.4, 0.6, 0.8, 1.0])
ARGS = ARGS_ENTITY_LEVEL_BASIC_CONCURRENT

sys.path.append("../../attack_generation/process_wikidata")
import wikidata_tools
sys.path.append("../../tools")
import adver_pipeline
import adver_tools

# DATA_FILE_PATH = "../../data/" + DATASET_NAME + "/" + DATASET_TYPE + ".txt"
ONTOROCK_E_PATH = "../../OntoRock/" + DATASET_TYPE + "/OntoRock-E/"
CLASS_DATA_PATH = "../../attack_generation/process_wikidata" + DATASET_NAME + "/" + DATASET_TYPE + "/class_data/"
PROCESSED_CLASS_INSTANCES = "../../attack_generation/process_wikidata" + DATASET_NAME + "/" + DATASET_TYPE \
                            + "/unseen_flair_processed_class_instances/"
MODEL_FILE_FOLDER_NAME = MODEL + "_entity_" + str(ARGS['attack_num']) + "attacks_concurrent_unseen_deleted"
OUTPUT_MAIN_PATH = "../../rockner/" + DATASET_NAME + "/" + DATASET_TYPE + "/" + MODEL_FILE_FOLDER_NAME + "/"
JSONL_FILES_PATH = OUTPUT_MAIN_PATH + "jsonl_files/"
RESULTS_PATH = OUTPUT_MAIN_PATH + "results/"
NONE_DATA_SAMPLES_PATH = OUTPUT_MAIN_PATH + "none_data_samples/"
ROBUSTNESS_PATH = OUTPUT_MAIN_PATH + "robustness/"
VISUALIZED_RESULTS_PATH = OUTPUT_MAIN_PATH + "visualized_results/"
F1_REPORTS_PATH = OUTPUT_MAIN_PATH + "f1_reports/"

ENTITY_TYPE_LIST_WITHOUT_PERSON = {"ontonotes_names": ['GPE', 'ORG', 'FAC', 'LOC', 'NORP', 'LAW', 'EVENT',
                                                       'WORK_OF_ART', 'PRODUCT', 'LANGUAGE'],
                                   "conll2003": ["ORG", "LOC"],
                                   "ontonotes_english": ['GPE', 'ORG', 'FAC', 'LOC', 'NORP', 'LAW', 'EVENT',
                                                         'WORK_OF_ART', 'PRODUCT', 'LANGUAGE']
                                   }
PRED_STAT_TYPE_LIST = {"ontonotes_english": ['ORG', 'FAC', 'LOC', 'GPE', 'NORP', 'EVENT', 'LAW',
                                             'WORK_OF_ART', 'PRODUCT', 'LANGUAGE', 'PERSON']}
STATUS = ["Samespan_Correct", "1_Correct", "2_Correct", "3_Correct", "Samespan_Wrong", "1_Wrong", "2_Wrong",
          "3_Wrong", "None"]

adver_tools.ENTITY_TYPE_LIST = adver_tools.ENTITY_TYPE_LIST_BY_DATASET_NAME[DATASET_NAME]
adver_pipeline.ENTITY_TYPE_LIST_WITHOUT_PERSON = ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]
wikidata_tools.ENTITY_TYPE_LIST_WITHOUT_PERSON = ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]


def load_adver_entities_predictions(path, attack_round):
    adver_prediction = {"entities": [], "predictions": [], "shifted_non_sample_entities": [], "sampled_entities": []}
    with open(path + str(attack_round) + '.jsonl') as f:
        for line in f:
            item = dict(json.loads(line.strip()))
            if not item['adver']['entities']:
                adver_prediction["entities"].append([])
                adver_prediction["sampled_entities"].append([])
            else:
                adver_prediction["entities"].append([[ent[1], ent[2], ent[3]] for ent in item["adver"]["entities"]])
                adver_prediction["sampled_entities"].append([[ent[1], ent[2], ent[3]]
                                                             for ent in item["original"]["sampled_entities"]])
            adver_prediction["predictions"].append(item["predictions"])
            if not item["adver"]["shifted_non_sample_entities"]:
                adver_prediction["shifted_non_sample_entities"].append([])
            else:
                adver_prediction["shifted_non_sample_entities"].append(item["adver"]["shifted_non_sample_entities"])
    return adver_prediction


def classify_entity_prediction(entity, predictions):
    if not predictions:
        return ["None"], ["None"]
    if entity in predictions:
        return ["Samespan_Correct"], [entity[2]]
    elif any(ent[0] == entity[0] and ent[1] == entity[1] for ent in predictions):
        for pred in predictions:
            if pred[0] == entity[0] and pred[1] == entity[1]:
                return ["Samespan_Wrong"], [pred[2]]
    else:
        ol_predictions = find_overlap_predictions(entity, predictions)
        if not ol_predictions:
            return ["None"], ["None"]
        ent_idx = set(range(entity[0], entity[1]+1))
        results = []
        pred_types = []
        for ol_pred in ol_predictions:
            ol_pred_idx = set(range(ol_pred[0], ol_pred[1]+1))
            d = min(3, len((ent_idx | ol_pred_idx) - (ent_idx & ol_pred_idx)))
            type_status = "Correct" if ol_pred[2] == entity[2] else "Wrong"
            results.append(str(d)+'_'+type_status)
            pred_types.append(ol_pred[2])
        return results, pred_types


def find_overlap_predictions(entity, predictions):
    ol_predictions = []
    # ol_pred_idx = set()
    for prediction in predictions:
        if max(0, (min(entity[1], prediction[1]) - max(entity[0], prediction[0]))) > 0:
            ol_predictions.append(prediction)
            # ol_pred_idx.update(set(range(prediction[0], prediction[1]+1)))
    return ol_predictions



def analyze_predictions(adver_ents_predictions):
    counters = {status: 0 for status in STATUS}
    unchanged_counters = {status: 0 for status in STATUS}
    changed_entity_num = 0
    unchanged_ent_num = 0
    norp_to_gpe = {'ratio': 0, 'num': 0}
    original_norp_num = 0
    pred_stat = {ent_type: {'ORG': 0, 'FAC': 0, 'LOC': 0, 'GPE': 0, 'NORP': 0, 'EVENT': 0, 'LAW': 0,
                            'WORK_OF_ART': 0, 'PRODUCT': 0, 'LANGUAGE': 0, 'PERSON': 0, 'None': 0}
                 for ent_type in PRED_STAT_TYPE_LIST[DATASET_NAME]}
    all_ents = adver_ents_predictions["entities"]
    all_predictions = adver_ents_predictions["predictions"]
    all_unchange_ents = adver_ents_predictions["shifted_non_sample_entities"]
    all_sampled_ents = adver_ents_predictions["sampled_entities"]
    for ents, predictions, unchanged_ents, sampled_ents in \
            zip(all_ents, all_predictions, all_unchange_ents, all_sampled_ents):
        changed_entity_num += len(ents)
        unchanged_ent_num += len(unchanged_ents)
        for entity, sampled_entity in zip(ents, sampled_ents):
            if sampled_entity[2] == "NORP":
                original_norp_num += 1
                if entity[2] == "GPE":
                    norp_to_gpe['num'] += 1
            class_status, pred_types = classify_entity_prediction(entity, predictions)
            for status in class_status:
                counters[status] += float(1/len(class_status))
            for pred_type in pred_types:
                if pred_type not in PRED_STAT_TYPE_LIST[DATASET_NAME]:
                    pred_type = "None"
                pred_stat[entity[2]][pred_type] += float(1/len(pred_types))
        for unchanged_entity in unchanged_ents:
            unchanged_class_status, pred_types = classify_entity_prediction(unchanged_entity, predictions)
            for unchanged_status in unchanged_class_status:
                unchanged_counters[unchanged_status] += float(1/len(unchanged_class_status))
            for pred_type in pred_types:
                if pred_type not in PRED_STAT_TYPE_LIST[DATASET_NAME]:
                    pred_type = "None"
                pred_stat[unchanged_entity[2]][pred_type] += float(1/len(pred_types))
    norp_to_gpe['ratio'] = 0 if original_norp_num == 0 else float(norp_to_gpe['num']/original_norp_num)
    return counters, changed_entity_num, unchanged_counters, unchanged_ent_num, pred_stat, norp_to_gpe


def output_robustness(path, ratio, counters, changed_entity_num, norp_to_gpe, attack_round):
    with open(path + str(attack_round) + ".tsv", "w+") as f:  #ROBUSTNESS
        for status in STATUS:
            f.write(status + ',' + str(ratio[status]) + '%'
                    + ',' + str(counters[status]) + '\n')
        f.write("changed_entity_num" + ',' + str(changed_entity_num) + '\n')
        f.write("NORP_to_GPE," + str(norp_to_gpe['num']) + "," + str(round(100*norp_to_gpe['ratio'], 2)) + "%")


def output_unchanged_robustness(path, ratio, uc_counters, uc_entity_num, attack_round):
    with open(path + str(attack_round) + ".unchanged.tsv", "w+") as f:  #ROBUSTNESS
        for status in STATUS:
            f.write(status + ',' + str(ratio[status]) + '%'
                    + ',' + str(uc_counters[status]) + '\n')
        f.write("unchanged_entity_num" + ',' + str(uc_entity_num) + '\n')


def output_pred_stat(path, pred_stat, attack_round):
    output_pred_stat = deepcopy(pred_stat)
    for ent_type in PRED_STAT_TYPE_LIST[DATASET_NAME]:
        output_pred_stat[ent_type]["total_num"] = sum(pred_stat[ent_type].values())
        for k, v in pred_stat[ent_type].items():
            if k == "total_num":
                continue
            output_pred_stat[ent_type][k+" (%)"] = str(round(100*float(output_pred_stat[ent_type][k]/output_pred_stat[ent_type]['total_num']), 2))
    df = pd.DataFrame(output_pred_stat, [k+" (%)" for k in PRED_STAT_TYPE_LIST[DATASET_NAME]]+["None (%)", "total_num"],
                      columns=PRED_STAT_TYPE_LIST[DATASET_NAME])
    df.T.to_csv(path + str(attack_round) + ".pred_stat.csv")


def analyze_robustness(attack_round):
    adver_ents_predictions = load_adver_entities_predictions(JSONL_FILES_PATH, attack_round)
    counters, changed_entity_num, unchg_cntrs, unchg_ent_num, pred_stat, norp_to_gpe = \
        analyze_predictions(adver_ents_predictions)
    ratio = {k: round(100*float(v/sum(counters.values())), 2) for k, v in counters.items()}
    output_robustness(ROBUSTNESS_PATH, ratio, counters, changed_entity_num, norp_to_gpe, attack_round)
    unchg_ratio = {k: round(100*float(v/sum(unchg_cntrs.values())), 2) for k, v in unchg_cntrs.items()}
    output_unchanged_robustness(ROBUSTNESS_PATH, unchg_ratio, unchg_cntrs, unchg_ent_num, attack_round)
    output_pred_stat(ROBUSTNESS_PATH, pred_stat, attack_round)


def model_eval_concurrent(attack_round):
    adver_pipeline.model_init(MODEL, model_path=MODEL_PATH, device="cuda:0")
    # random.seed(ARGS['seeds'][attack_round])
    # sample_pct = ARGS['sampled_pct'][attack_round]
    all_sents, all_tags = adver_tools.read_data(ONTOROCK_E_PATH + str(attack_round) + ".txt")
    # class_instances = wikidata_tools.load_class_instances(PROCESSED_CLASS_INSTANCES)
    # class_instances["PERSON"] = wikidata_tools.load_person_instances(PROCESSED_CLASS_INSTANCES)

    jsonl_data = load_jsonl_files(ONTOROCK_E_PATH, attack_round)
    # sampled_entities_by_sid = sample_entities(entities_by_sid, sample_pct, ent_classes, class_instances)

    all_gt_tags = []
    all_predicted_tags = []
    all_adver_sents = []
    all_predictions = []
    for sent_id, sent in tqdm(enumerate(all_sents), total=len(all_sents)):
        gt_tags = all_tags[sent_id]
        all_gt_tags.append(gt_tags)
        predictions = adver_pipeline.predict_sentence(sent, MODEL)
        pred_tags = adver_tools.get_all_tags(sent, predictions)
        all_predicted_tags.append(pred_tags)
        all_adver_sents.append(sent)
        all_predictions.append(predictions)
    output_results_concurrent(RESULTS_PATH, all_adver_sents, all_gt_tags, all_predicted_tags, attack_round)
    output_jsonl_files_concurrent(JSONL_FILES_PATH, jsonl_data, all_predictions, attack_round)
    # output_none_data_samples(NONE_DATA_SAMPLES_PATH, all_none_sampled_entities)


# def sample_entities(entities_by_sid, pct, ent_class, class_instances):
#     possible_ents_by_sid = {}
#     possible_ents_set = set()
#     for sent_id, ents in entities_by_sid.items():
#         possible_ents_by_sid[sent_id] = []
#         for ent in ents:
#             if ent[3] == "PERSON":
#                 possible_ents_by_sid[sent_id].append(ent)
#                 possible_ents_set.add(ent)
#                 continue
#             if tuple(ent) in ent_class[ent[3]].keys():
#                 classes = ent_class[ent[3]][tuple(ent)]['classes']
#                 classes_with_instances = {k: v for k, v in classes.items() if k in class_instances[ent[3]].keys()
#                                           and class_instances[ent[3]][k]['instance_num'] != 0}
#                 if classes_with_instances:
#                     possible_ents_by_sid[sent_id].append(ent)
#                     possible_ents_set.add(ent)
#
#     sampled_entities_set = random.sample(possible_ents_set, k=int(round(pct*len(possible_ents_set), 0)))
#     sampled_entities_by_sid = {}
#     for sent_id, entitites in possible_ents_by_sid.items():
#         sampled_entities_by_sid[sent_id] = [ent for ent in entitites if ent in sampled_entities_set]
#     return sampled_entities_by_sid


def output_results_concurrent(path, all_adver_sents, all_gt_tags, all_pred_tags, attack_round):
    with open(path + str(attack_round) + '.results', 'w+') as f:
        for sent, gt_tags, pred_tags in zip(all_adver_sents, all_gt_tags, all_pred_tags):
            for token, tag, pred_tag in zip(sent, gt_tags, pred_tags):
                # token     correct_tag     predicted_tag
                f.write(token + '\t' + tag + '\t' + pred_tag + '\n')
            f.write('\n')


def output_jsonl_files_concurrent(path, all_adver_lists, all_predictions, attack_round):
    with open(path + str(attack_round) + '.jsonl', 'w+') as f:
        for sent_id, adver_list in enumerate(all_adver_lists):
            if adver_list:
                predictions = all_predictions[sent_id]
                # adver_entities = [[ent[1], ent[2], ent[3]] for ent in adver_list["adver"]["entities"]]
                line = json.dumps(dict(original=adver_list["original"], adver=adver_list["adver"],
                                       entities_with_derivation=adver_list["entities_with_derivation"],
                                       predictions=predictions))
                f.write(line + '\n')


def load_jsonl_files(path, attack_round):
    jsonl_data = []
    with open(path + str(attack_round) + '.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            jsonl_data.append(dict(original=data['original'],
                                   adver=data['adver'],
                                   entities_with_derivation=data['entities_with_derivation']))
    return jsonl_data


def main():
    attack_round = int(sys.argv[1])

    if not os.path.exists(OUTPUT_MAIN_PATH) or not os.path.exists(RESULTS_PATH):
        os.system("mkdir -p " + " ".join([JSONL_FILES_PATH, RESULTS_PATH, ROBUSTNESS_PATH, VISUALIZED_RESULTS_PATH, F1_REPORTS_PATH]))

    model_eval_concurrent(attack_round)

    time.sleep(10)

    analyze_robustness(attack_round)



if __name__ == '__main__':
    main()

