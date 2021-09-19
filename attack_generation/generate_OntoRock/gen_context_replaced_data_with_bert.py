import random
from tqdm import tqdm
import sys
import nltk
from happytransformer import HappyROBERTA


sys.path.append("../process_wikidata")

sys.path.append("../../tools")
import adver_pipeline
import adver_tools

DATASET_NAME = "ontonotes_english"
DATASET_TYPE = "dev"
DATA_FILE_PATH = "../../data/" + DATASET_NAME + "/" + DATASET_TYPE + ".txt"
MODEL = "blstm_crf"
MODEL_PATH = "../../models/model_training/resources/taggers/" + MODEL
ARGS_ENTITY_LEVEL_BASIC = dict(attack_num=5, seeds=[1, 2, 3, 4, 5])
ARGS = ARGS_ENTITY_LEVEL_BASIC
# MODEL_FILE_FOLDER_NAME = MODEL + "_entity_" + str(ARGS['attack_num']) + "attacks_concurrent_unseen_deleted"
USE_FILTER = False
REPLACE_MARK = "context"  # both or context
ONTOROCK_E_PATH = "../../OntoRock/" + DATASET_TYPE + "/OntoRock-E/"

if REPLACE_MARK == "context":
    OUTPUT_PATH = "../../OntoRock/" + DATASET_TYPE + "/OntoRock-C/"
else:
    OUTPUT_PATH = "../../OntoRock/" + DATASET_TYPE + "/OntoRock-F/"

if USE_FILTER:
    CONTEXT_REPLACED_DATA_PATH = OUTPUT_PATH + "filtered.txt"
else:
    CONTEXT_REPLACED_DATA_PATH = OUTPUT_PATH + "unfiltered.txt"


POS_TAGS_LIST = ["JJ", "JJR", "JJS", "NN", "NNP", "NNS", "RB", "RBR", "RBS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]

mlm_model = HappyROBERTA("roberta-base")


def create_masks(sent, entities, k=5):
    masked_sents_str = []
    entity_idxes = set()
    for ent in entities:
        entity_idxes.update(set(range(ent[1], ent[2]+1)))
    pos_tags = nltk.pos_tag(sent)
    candidate_idxes = list({idx for idx, pos in enumerate(pos_tags) if pos[1] in POS_TAGS_LIST} - entity_idxes)
    if len(candidate_idxes) == 0:
        return [" ".join(sent)]
    for _ in range(k):
        num_masks = random.randint(1, min(3, len(candidate_idxes)))
        mask_idxes = random.sample(candidate_idxes, k=num_masks)
        masked_sent = [token if token_idx not in mask_idxes else "<mask>" for token_idx, token in enumerate(sent)]
        masked_sents_str.append(" ".join(masked_sent))
    return masked_sents_str


def fill_masks(masked_sents_str):
    replaced_sents = []
    all_masked_idxes = []
    for maksed_sent_str in masked_sents_str:
        num_masks = sum([t == "<mask>" for t in maksed_sent_str.split()])
        replaced_sent_str = maksed_sent_str
        masked_idxes = [idx for idx, token in enumerate(maksed_sent_str.split()) if token == "<mask>"]
        for _ in range(num_masks):
            results = mlm_model.predict_mask(replaced_sent_str, num_results=100)[50:]
            filtered_results = [result for result in results if all(ord(char) < 128 for char in result["word"])]
            choice = random.choice(filtered_results)["word"]
            replaced_sent_str = replaced_sent_str.replace("<mask>", choice, 1)
        replaced_sents.append(replaced_sent_str.split())
        all_masked_idxes.append(masked_idxes)
    return replaced_sents, all_masked_idxes


def filter_sents_with_model(replaced_sents, all_masked_idxes, entities):
    num_worong_pred = [0] * len(replaced_sents)
    for sent_idx, replaced_sent in enumerate(replaced_sents):
        if USE_FILTER:
            predictions = adver_pipeline.predict_sentence(replaced_sent, MODEL)
        else:
            predictions = []
        wrong_pred_ents = [ent for ent in entities if ent[1:] not in predictions]
        num_worong_pred[sent_idx] = len(wrong_pred_ents)
    max_wrong_pred_num = max(num_worong_pred)
    max_worng_pred_idxes = [idx for idx, num in enumerate(num_worong_pred) if num == max_wrong_pred_num]
    sampled_sent_idx = random.sample(max_worng_pred_idxes, k=1)[0]
    result = replaced_sents[sampled_sent_idx]
    result_masked_idxes = all_masked_idxes[sampled_sent_idx]
    return result, result_masked_idxes, max_wrong_pred_num


def replace_context(all_sents, entities_by_sid):
    context_replaced_sents = []
    max_wrong_ratio = []
    all_sents_masked_idxes = []
    for sent_id, sent in enumerate(tqdm(all_sents)):
        masked_sents_str = create_masks(sent, entities_by_sid[sent_id])
        replaced_sents, all_masked_idxes = fill_masks(masked_sents_str)
        result_sent, result_masked_idxes, max_wrong_pred_num = \
            filter_sents_with_model(replaced_sents, all_masked_idxes, entities_by_sid[sent_id])
        context_replaced_sents.append(result_sent)
        all_sents_masked_idxes.append(result_masked_idxes)
        max_wrong_ratio.append(0 if max_wrong_pred_num == 0
                               else round(float(max_wrong_pred_num/len(entities_by_sid[sent_id])), 2))
    return context_replaced_sents, all_sents_masked_idxes, max_wrong_ratio


def output_context_replaced_sents(path, context_replaced_sents, all_tags):
    with open(path, 'w+') as f:
        for sent, tags in zip(context_replaced_sents, all_tags):
            for token, tag in zip(sent, tags):
                f.write(token + '\t' + tag + '\n')
            f.write('\n')


def output_log(path, all_sents, context_replaced_sents, all_sents_masked_idxes, max_wrong_ratio):
    with open(path[:-4] + '_log.txt', 'w+') as f:
        for ori_sent, rpl_sent, masked_idxes, ratio in zip(all_sents[:len(context_replaced_sents)],
                                                           context_replaced_sents,
                                                           all_sents_masked_idxes,
                                                           max_wrong_ratio):
            ori_sent = [token if idx not in masked_idxes
                        else "<" + token + ">" for idx, token in enumerate(ori_sent)]
            rpl_sent = [token if idx not in masked_idxes
                        else "<" + token + ">" for idx, token in enumerate(rpl_sent)]
            f.write(" ".join(ori_sent) + "\n" + " ".join(rpl_sent) + "\n" + str(ratio) + "\n\n")


def main():
    if REPLACE_MARK == "context":
        random.seed(42)
        all_sents, all_tags = adver_tools.read_data(DATA_FILE_PATH)
        entities_by_type, entities_by_sid, entities_set = adver_tools.update_NER_dict(all_sents, all_tags)
    else:
        attack_round = 4
        random.seed(ARGS['seeds'][attack_round])
        all_sents, all_tags = adver_tools.read_data(ONTOROCK_E_PATH + str(attack_round) + ".txt")
        entities_by_type, entities_by_sid, entities_set = adver_tools.update_NER_dict(all_sents, all_tags)

    if USE_FILTER:
        adver_pipeline.model_init(MODEL, model_path=MODEL_PATH, device="cuda:0")

    context_replaced_sents, all_sents_masked_idxes, max_wrong_ratio = replace_context(all_sents, entities_by_sid)

    output_context_replaced_sents(CONTEXT_REPLACED_DATA_PATH, context_replaced_sents, all_tags)

    output_log(CONTEXT_REPLACED_DATA_PATH, all_sents, context_replaced_sents, all_sents_masked_idxes,
               max_wrong_ratio)


if __name__ == '__main__':
    main()

