import blink.main_dense as main_dense
import argparse
import sys
import json

sys.path.append("../../tools")
import adver_tools

DATASET_NAME = "ontonotes_english"
DATASET_TYPE = "train"
DATA_FILE_PATH = "../data/" + DATASET_NAME + "/" + DATASET_TYPE + ".txt"
ENTITY_TYPE_LIST_WITHOUT_PERSON = {"ontonotes_names": ['GPE', 'ORG', 'FAC', 'LOC', 'NORP', 'LAW', 'EVENT',
                                                       'WORK_OF_ART', 'PRODUCT', 'LANGUAGE'],
                                   "conll2003": ["ORG", "LOC"],
                                   "ontonotes_english": ['GPE', 'ORG', 'FAC', 'LOC', 'NORP', 'LAW', 'EVENT',
                                                         'WORK_OF_ART', 'PRODUCT', 'LANGUAGE']
                                   }

models_path = "../../BLINK/models/"

config = {
    "test_entities": None,
    "test_mentions": None,
    "interactive": False,
    "top_k": 10,
    "biencoder_model": models_path + "biencoder_wiki_large.bin",
    "biencoder_config": models_path + "biencoder_wiki_large.json",
    # "biencoder_params": {"eval_batch_size": 50},
    "entity_catalogue": models_path + "entity.jsonl",
    "entity_encoding": models_path + "all_entities_large.t7",
    "crossencoder_model": models_path + "crossencoder_wiki_large.bin",
    "crossencoder_config": models_path + "crossencoder_wiki_large.json",
    "fast": False,
    "output_path": "../../BLINK/output/",
    "faiss_index": "flat",
    "index_path": models_path + "faiss_flat_index.pkl"
}


def load_data_to_link(all_sents, entities_by_sid):
    data_to_link = []
    data_to_link_entities = []
    id_counter = 0
    for s_id, sent in enumerate(all_sents):
        for ent in entities_by_sid[s_id]:
            entity = [ent[1], ent[2], ent[3]]
            if entity[2] in ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]:
                data_to_link.append(dict(id=id_counter,
                                         label="unknown",
                                         label_id=-1,
                                         context_left=" ".join(sent[:entity[0]]).lower()
                                         if entity[1] - entity[0] + 1 <= 25 else "",
                                         mention=" ".join(sent[entity[0]:entity[1] + 1]).lower()
                                         if entity[1] - entity[0] + 1 <= 25
                                         else "Law of the People's Republic of China on Safeguarding National Security in the Hong Kong Special Administrative Region".lower(),
                                         context_right=" ".join(sent[entity[1] + 1:]).lower()
                                         if entity[1] - entity[0] + 1 <= 25 else ""
                                         ))
                data_to_link_entities.append(ent)
                id_counter += 1
    return data_to_link, data_to_link_entities


def output_predictions(predictions, scores, data_to_link_entities, start_idx, end_idx):
    with open("./" + DATASET_NAME + "/" + DATASET_TYPE + "/titles/blink_results."
              + str(start_idx) + "_" + str(end_idx) + ".jsonl", 'w+') as f:
        for result_id, ent in enumerate(data_to_link_entities):
            prediction = predictions[result_id]
            score = scores[result_id]
            f.write(json.dumps(dict(entity=ent, prediction=prediction, score=[str(s) for s in score])) + '\n')


def main():
    ###
    start_idx = int(sys.argv[1])
    ###
    all_sents, all_tags = adver_tools.read_data(DATA_FILE_PATH)
    entities_by_type, entities_by_sid, entities_set = adver_tools.update_NER_dict(all_sents, all_tags)
    data_to_link, data_to_link_entities = load_data_to_link(all_sents, entities_by_sid)
    # print(len(data_to_link))
    ###
    end_idx = min(start_idx+3000, len(data_to_link))
    ###
    # sample_data_to_link = data_to_link[33000:36000]
    # sample_data_to_link_entities = data_to_link_entities[33000:36000]
    # counters = {"context_left": 0, "mention": 0, "context_right": 0}
    # long_sents = {}
    # for i in range(0, 39000):
    #     sent = all_sents[i]
    #     if len(sent) >= 32:
    #         long_sents[i] = sent
    #     context_left = data_to_link[i]['context_left']
    #     mention = data_to_link[i]['mention']
    #     context_right = data_to_link[i]['context_right']
    #     if context_left == '':
    #         counters['context_left'] += 1
    #     if context_left == '':
    #         counters['context_right'] += 1
    #     if mention == '':
    #         counters['mention'] += 1
    #     if context_left == mention == context_right == '':
    #         print("NONE", data_to_link_entities[i])
    # print(counters)


    args = argparse.Namespace(**config)

    models = main_dense.load_models(args, logger=None)

    _, _, _, _, _, predictions, scores, = main_dense.run(args, None, *models,
                                                         test_data=data_to_link[start_idx:end_idx])

    print(len(predictions), len(scores))
    output_predictions(predictions, scores, data_to_link_entities[start_idx:end_idx], start_idx, end_idx)


if __name__ == '__main__':
    main()
