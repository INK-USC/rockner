import random
import sys
import json
from tqdm import tqdm

DATASET_NAME = "ontonotes_english"
DATASET_TYPE = "test"
ARGS_ENTITY_LEVEL_BASIC_CONCURRENT = dict(attack_num=5, seeds=[0, 1, 2, 3, 4], sampled_pct=[0.2, 0.4, 0.6, 0.8, 1.0])
ARGS = ARGS_ENTITY_LEVEL_BASIC_CONCURRENT

sys.path.append("../process_wikidata/")
import wikidata_tools
sys.path.append("../../tools")
import adver_pipeline
import adver_tools

DATA_FILE_PATH = "../../data/" + DATASET_NAME + "/" + DATASET_TYPE + ".txt"
CLASS_DATA_PATH = "../process_wikidata/" + DATASET_NAME + "/" + DATASET_TYPE + "/class_data/"
PROCESSED_CLASS_INSTANCES = "../process_wikidata/" + DATASET_NAME + "/" + DATASET_TYPE \
                            + "/unseen_flair_processed_class_instances/"
OUTPUT_PATH = "../../OntoRock/" + DATASET_TYPE + "/OntoRock-E/"
# MODEL_FILE_FOLDER_NAME = MODEL + "_entity_" + str(ARGS['attack_num']) + "attacks_concurrent_unseen_deleted"

# JSONL_FILES_PATH = OUTPUT_MAIN_PATH + "jsonl_files/"
# RESULTS_PATH = OUTPUT_MAIN_PATH + "results/"
# NONE_DATA_SAMPLES_PATH = OUTPUT_MAIN_PATH + "none_data_samples/"
# ROBUSTNESS_PATH = OUTPUT_MAIN_PATH + "robustness/"
# VISUALIZED_RESULTS_PATH = OUTPUT_MAIN_PATH + "visualized_results/"
# F1_REPORTS_PATH = OUTPUT_MAIN_PATH + "f1_reports/"

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


def gen_ontorock_c(attack_round):
    random.seed(ARGS['seeds'][attack_round])
    sample_pct = ARGS['sampled_pct'][attack_round]
    all_sents, all_tags = adver_tools.read_data(DATA_FILE_PATH)
    entities_by_type, entities_by_sid, entities_set = adver_tools.update_NER_dict(all_sents, all_tags)
    ent_classes = wikidata_tools.load_ent_classes_dict(CLASS_DATA_PATH)
    class_ents = wikidata_tools.load_class_ents_dict(CLASS_DATA_PATH)
    class_instances = wikidata_tools.load_class_instances(PROCESSED_CLASS_INSTANCES)
    class_instances["PERSON"] = wikidata_tools.load_person_instances(PROCESSED_CLASS_INSTANCES)

    sampled_entities_by_sid = sample_entities(entities_by_sid, sample_pct, ent_classes, class_instances)

    all_gt_tags = []
    # all_predicted_tags = []
    all_adver_sents = []
    all_adver_lists = []
    # all_predictions = []
    for sent_id, sent in tqdm(enumerate(all_sents), total=len(all_sents)):
        if sent_id in sampled_entities_by_sid.keys():
            sampled_entities = sorted(sampled_entities_by_sid[sent_id], key=lambda x: x[1])
        else:
            sampled_entities = []
        if len(entities_by_sid[sent_id]) == 0 or not sampled_entities:
            gt_tags = all_tags[sent_id]
            all_gt_tags.append(gt_tags)
            all_adver_sents.append(sent)
            adver_list = dict(original=dict(sentence=" ".join(sent),
                                            all_entities=entities_by_sid[sent_id],
                                            sampled_entities=[]),
                              adver=dict(sentence=" ".join(sent),
                                         entities=[],
                                         shifted_non_sample_entities=[[ent[1], ent[2], ent[3]] for ent in
                                                                      entities_by_sid[sent_id]]),
                              entities_with_derivation="")
            all_adver_lists.append(adver_list)
        else:
            complete_ents = entities_by_sid[sent_id]
            complete_ents = sorted(complete_ents, key=lambda x: x[1], reverse=False)
            adver_sentence, adver_entities, adver_list, shifted_non_sample_entities = adver_pipeline. \
                entity_level_replace_with_homogeneous_wikiinstances_concurrent(sent, complete_ents, sampled_entities,
                                                                               ent_classes, class_ents, class_instances,
                                                                               ARGS)
            if shifted_non_sample_entities:
                gt_entities = adver_entities + shifted_non_sample_entities
                gt_tags = adver_tools.get_all_tags(adver_sentence, gt_entities)
            else:
                gt_tags = adver_tools.get_all_tags(adver_sentence, adver_entities)
            # all_predicted_tags.append(pred_tags)
            all_gt_tags.append(gt_tags)
            all_adver_sents.append(adver_sentence)
            all_adver_lists.append(adver_list)
            # all_predictions.append(predictions)
    output_results_concurrent(OUTPUT_PATH, all_adver_sents, all_gt_tags, attack_round)
    output_jsonl_files_concurrent(OUTPUT_PATH, all_adver_lists, attack_round)


def sample_entities( entities_by_sid, pct, ent_class, class_instances ):
    possible_ents_by_sid = {}
    possible_ents_set = set()
    for sent_id, ents in entities_by_sid.items():
        possible_ents_by_sid[sent_id] = []
        for ent in ents:
            if ent[3] == "PERSON":
                possible_ents_by_sid[sent_id].append(ent)
                possible_ents_set.add(ent)
                continue
            if tuple(ent) in ent_class[ent[3]].keys():
                classes = ent_class[ent[3]][tuple(ent)]['classes']
                classes_with_instances = {k: v for k, v in classes.items() if k in class_instances[ent[3]].keys()
                                          and class_instances[ent[3]][k]['instance_num'] != 0}
                if classes_with_instances:
                    possible_ents_by_sid[sent_id].append(ent)
                    possible_ents_set.add(ent)

    sampled_entities_set = random.sample(possible_ents_set, k=int(round(pct * len(possible_ents_set), 0)))
    sampled_entities_by_sid = {}
    for sent_id, entitites in possible_ents_by_sid.items():
        sampled_entities_by_sid[sent_id] = [ent for ent in entitites if ent in sampled_entities_set]
    return sampled_entities_by_sid


def output_results_concurrent(path, all_adver_sents, all_gt_tags, attack_round):
    with open(path + str(attack_round) + '.txt', 'w+') as f:
        for sent, gt_tags in zip(all_adver_sents, all_gt_tags):
            for token, tag in zip(sent, gt_tags):
                # token     correct_tag     predicted_tag
                f.write(token + '\t' + tag + '\n')
            f.write('\n')


def output_jsonl_files_concurrent(path, all_adver_lists, attack_round):
    with open(path + str(attack_round) + '.jsonl', 'w+') as f:
        for sent_id, adver_list in enumerate(all_adver_lists):
            if adver_list:
                line = json.dumps(dict(original=adver_list["original"], adver=adver_list["adver"],
                                       entities_with_derivation=adver_list["entities_with_derivation"]))
                f.write(line + '\n')


def main():
    attack_round = int(sys.argv[1])

    gen_ontorock_c(attack_round)


if __name__ == '__main__':
    main()
