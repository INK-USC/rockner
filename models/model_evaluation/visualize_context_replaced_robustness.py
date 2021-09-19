import sys

DATASET_NAME = "ontonotes_english"
DATASET_TYPE = "test"
MODEL = "roberta_crf"
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
# VISUALIZED_RESULTS_PATH = OUTPUT_MAIN_PATH + "/visualized_results/"
USE_FILTER = True
REPLACE_MARK = "context"  # both or context

MODEL_FILE_FOLDER_NAME = MODEL + "_entity_" + str(ARGS['attack_num']) + "attacks_concurrent_unseen_deleted"
OUTPUT_MAIN_PATH = "../../rockner/" + DATASET_NAME + "/" + DATASET_TYPE + "/" + MODEL_FILE_FOLDER_NAME + "/"
CONTEXT_REPLACED_ROBUSTNESS_PATH = OUTPUT_MAIN_PATH + "context_replaced_robustness/"

if USE_FILTER:
    CONTEXT_REPLACED_DATA_PATH = "../../data/" + DATASET_NAME + "/" + DATASET_TYPE \
                                 + "." + REPLACE_MARK + "_replaced_filtered.txt"
    ROBUSTNESS_PATH = OUTPUT_MAIN_PATH + "context_replaced_robustness/" \
                      + REPLACE_MARK + "_replaced_filtered"
else:
    CONTEXT_REPLACED_DATA_PATH = "../../data/" + DATASET_NAME + "/" + DATASET_TYPE \
                                 + "." + REPLACE_MARK + "_replaced.txt"
    ROBUSTNESS_PATH = OUTPUT_MAIN_PATH + "context_replaced_robustness/" \
                      + REPLACE_MARK + "_replaced"

if REPLACE_MARK == "context":
    DATA_FILE_PATH = OUTPUT_MAIN_PATH + DATASET_TYPE + ".results"
else:
    DATA_FILE_PATH = OUTPUT_MAIN_PATH + "results/4.results"


ENTITY_TYPE_LIST_WITHOUT_PERSON = {"ontonotes_names": ['GPE', 'ORG', 'FAC', 'LOC', 'NORP', 'LAW', 'EVENT',
                                                       'WORK_OF_ART', 'PRODUCT', 'LANGUAGE'],
                                   "conll2003": ["ORG", "LOC"],
                                   "ontonotes_english": ['GPE', 'ORG', 'FAC', 'LOC', 'NORP', 'LAW', 'EVENT',
                                                         'WORK_OF_ART', 'PRODUCT', 'LANGUAGE']
                                   }


adver_tools.ENTITY_TYPE_LIST = adver_tools.ENTITY_TYPE_LIST_BY_DATASET_NAME[DATASET_NAME]
adver_pipeline.ENTITY_TYPE_LIST_WITHOUT_PERSON = ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]
wikidata_tools.ENTITY_TYPE_LIST_WITHOUT_PERSON = ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]


def visualize_data(ori_all_sents, ori_all_tags, ori_all_pred_tags, ori_entities_by_sid,
                   rpl_all_sents, rpl_all_tags, rpl_all_pred_tags, rpl_entities_by_sid):
    counter = 0
    with open(ROBUSTNESS_PATH + "_visualized.txt", "w+") as f:
        f.write("Original sentence w/ ground truth.\nOriginal sentence w/ predictions.\n"
                + "Context replaced sentence w/ ground truth.\nContext replaced sentence w/ predictions.\n\n")
        for sent_id, (ori_sent, ori_tags, ori_preds, rpl_sent, rpl_tags, rpl_preds) \
                in enumerate(zip(ori_all_sents, ori_all_tags, ori_all_pred_tags,
                                 rpl_all_sents, rpl_all_tags, rpl_all_pred_tags)):
            if len(ori_sent) != len(rpl_sent):
                counter += 1
                continue
            ori_sent_diff = [token if token == rpl_sent[token_idx] else "{"+token+"}"
                                 for token_idx, token in enumerate(ori_sent)]
            ori_sent_diff_str = " ".join(ori_sent_diff)
            vis_gt_ori_sent_diff_str = make_visualized_sent(ori_sent_diff_str, ori_entities_by_sid[sent_id])
            ori_pred_ents = gen_predicted_entities(sent_id, ori_preds)
            vis_pred_ori_sent_diff_str = make_visualized_sent(ori_sent_diff_str, ori_pred_ents)
            ori_wrong_preds = [pred_ent for pred_ent in ori_pred_ents if tuple(pred_ent) not in ori_entities_by_sid[sent_id]]
            rpl_sent_diff = [token if token == ori_sent[token_idx] else "{"+token+"}"
                             for token_idx, token in enumerate(rpl_sent)]
            rpl_sent_diff_str = " ".join(rpl_sent_diff)
            vis_gt_rpl_sent_diff_str = make_visualized_sent(rpl_sent_diff_str, rpl_entities_by_sid[sent_id])
            rpl_pred_ents = gen_predicted_entities(sent_id, rpl_preds)
            vis_pred_rpl_sent_diff_str = make_visualized_sent(rpl_sent_diff_str, rpl_pred_ents)
            rpl_wrong_preds = [pred_ent for pred_ent in rpl_pred_ents if tuple(pred_ent) not in rpl_entities_by_sid[sent_id]]
            if not ori_wrong_preds and not rpl_wrong_preds:
                continue
            f.write("\n------------------------ SENT_ID: " + str(sent_id) + " ------------------------\n")
            f.write(vis_gt_ori_sent_diff_str + "\n" + vis_pred_ori_sent_diff_str + "\n")
            for ent in ori_wrong_preds:
                f.write(" ".join(ori_sent[ent[1]: ent[2]+1]) + "\n")
            f.write("\n")
            f.write(vis_gt_rpl_sent_diff_str + "\n" + vis_pred_rpl_sent_diff_str + "\n")
            for ent in rpl_wrong_preds:
                f.write(" ".join(rpl_sent[ent[1]: ent[2]+1]) + "\n")
            f.write("\n")
    print(counter)


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


def gen_predicted_entities(sent_id, pred_tags):
    pred_ents = []
    start_idx = 0
    end_idx = 0
    ent_type = ""
    for token_idx, tag in enumerate(pred_tags):
        if tag[0] == "B":
            start_idx = token_idx
            end_idx = token_idx
            ent_type = tag[2:]
            if token_idx == len(pred_tags)-1:
                pred_ents.append([sent_id, start_idx, end_idx, ent_type])
            elif pred_tags[token_idx+1][0] == "O" or pred_tags[token_idx+1][0] == "B":
                pred_ents.append([sent_id, start_idx, end_idx, ent_type])
        elif tag[0] == "I":
            end_idx = token_idx
            if token_idx == len(pred_tags)-1:
                pred_ents.append([sent_id, start_idx, end_idx, ent_type])
            elif pred_tags[token_idx+1][0] == "O" or pred_tags[token_idx+1][0] == "B":
                pred_ents.append([sent_id, start_idx, end_idx, ent_type])
    return pred_ents


def main():
    ori_all_sents, ori_all_tags, ori_all_pred_tags = adver_tools.read_results_data(DATA_FILE_PATH)
    _, ori_entities_by_sid, _ = adver_tools.update_NER_dict(ori_all_sents, ori_all_tags)

    rpl_all_sents, rpl_all_tags, rpl_all_pred_tags = adver_tools.read_results_data(ROBUSTNESS_PATH + ".results")
    _, rpl_entities_by_sid, _ = adver_tools.update_NER_dict(rpl_all_sents, rpl_all_tags)

    visualize_data(ori_all_sents, ori_all_tags, ori_all_pred_tags, ori_entities_by_sid,
                   rpl_all_sents, rpl_all_tags, rpl_all_pred_tags, rpl_entities_by_sid)



if __name__ == '__main__':
    main()
