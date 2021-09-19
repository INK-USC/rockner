import requests
import json
import wikidata_tools
import sys
from tqdm import tqdm
from urllib.parse import quote
from time import sleep


sys.path.append("../../tools")
import adver_tools


DATASET_NAME = "ontonotes_english"
DATASET_TYPE = "train"
DATA_FILE_PATH = "../data/" + DATASET_NAME + "/" + DATASET_TYPE + ".txt"
FREQ_FILE_PATH = "../../attack_generation/entity_statistics/" + DATASET_NAME + "/" + DATASET_TYPE
API_RESULTS_PATH = DATASET_NAME + "/" + DATASET_TYPE + '/api_results/'
# REACQUIRED_API_RESULTS_PATH = DATASET_NAME + "/" + DATASET_TYPE + '/reacquired_api_results/'
NONE_ID_ENTITY_PATH = DATASET_NAME + "/" + DATASET_TYPE + '/none_id_entities/'
ENTITY_WIKIDATA_PATH = DATASET_NAME + "/" + DATASET_TYPE + '/entity_wikidata_pair/'
BLINK_RESULTS = DATASET_NAME + "/" + DATASET_TYPE + "/titles/blink_results.jsonl"
ENTITY_TYPE_LIST_WITHOUT_PERSON = {"ontonotes_names": ['GPE', 'ORG', 'FAC', 'LOC', 'NORP', 'LAW', 'EVENT',
                                                       'WORK_OF_ART', 'PRODUCT', 'LANGUAGE'],
                                   "conll2003": ["ORG", "LOC"],
                                   "ontonotes_english": ['GPE', 'ORG', 'FAC', 'LOC', 'NORP', 'LAW', 'EVENT',
                                                         'WORK_OF_ART', 'PRODUCT', 'LANGUAGE']}


wikidata_tools.ENTITY_TYPE_LIST_WITHOUT_PERSON = ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]



###
# sample_number = 20
###


def load_linked_entities():
    entities_by_types = {ent_type: [] for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]}
    with open(BLINK_RESULTS, 'r') as f:
        for line in f:
            data = dict(json.loads(line.strip()))
            entity = data["entity"]  # len(entity) is 4
            title = data["prediction"][0]
            entity.append(title)  # len(entity) is 5
            entities_by_types[entity[3]].append(entity)
    return entities_by_types



def acquire_via_api_output(linked_entities, all_sents):
    counter = 0
    title_result_dict = {}
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]:
        entities_by_type = linked_entities[ent_type]
        # entity_result_dict = {}
        with open(API_RESULTS_PATH+ent_type+'.linked_wikiapi.jsonl', 'w+') as f:
            for ent in tqdm(entities_by_type):  # len(ent) is 5.
                if 3000 == counter:
                    counter = 0
                    sleep(300)
                if not ent[4] in title_result_dict.keys():
                    qent = quote(ent[4])
                    try:
                        result = requests.get('https://en.wikipedia.org/w/api.php?action=query&format=json&prop=pageprops&titles='+qent)
                    except Exception as e:
                        sleep(600)
                        result = requests.get('https://en.wikipedia.org/w/api.php?action=query&format=json&prop=pageprops&titles='+qent)
                    title_result_dict[ent[4]] = json.loads(result.text)
                    counter += 1
                    f.write(json.dumps(dict(entity=ent[:4], ent_text=" ".join(all_sents[ent[0]][ent[1]:ent[2]+1]),
                                            linked_title=ent[4], result=json.loads(result.text))) + '\n')
                    sleep(0.01)
                else:
                    f.write(json.dumps(dict(entity=ent[:4], ent_text=" ".join(all_sents[ent[0]][ent[1]:ent[2]+1]),
                                            linked_title=ent[4], result=title_result_dict[ent[4]])) + '\n')


def read_api_results(path):
    api_results = {}
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]:
        api_results[ent_type] = []
        with open(path + ent_type + '.linked_wikiapi.jsonl', 'r') as f:
            for line in f:
                api_results[ent_type].append(line.strip())
    return api_results


def process_results(api_results):
    none_id_entities = {}
    none_id_titles = {}
    entity_wikidata_dict = {}
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]:
        results = api_results[ent_type]
        none_id_entities[ent_type] = []
        none_id_titles[ent_type] = []
        entity_wikidata_dict[ent_type] = {}
        for r in results:
            data = json.loads(r)
            entity = data['entity']     # len(entity) is 4
            linked_title = data['linked_title']
            pages = dict(data['result']['query']['pages'])
            wikidata = wikidata_tools.get_wikidata_from_api_result(pages)
            if wikidata:
                entity_wikidata_dict[ent_type][tuple(entity)] = {"linked_title": linked_title, "wikidata": wikidata}
            else:
                none_id_entities[ent_type].append(entity)
                none_id_titles[ent_type].append(linked_title)
    return entity_wikidata_dict, none_id_entities, none_id_titles


def output_entity_wikidata_pair(entity_wikidata_dict, all_sents):
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]:
        with open(ENTITY_WIKIDATA_PATH + ent_type + '.linked_wikidata.jsonl', 'w+') as f:
            for entity, data in entity_wikidata_dict[ent_type].items():
                f.write(json.dumps(dict(entity=entity, ent_text=" ".join(all_sents[entity[0]][entity[1]:entity[2]+1]),
                                        linked_title=data['linked_title'], wikidata=data['wikidata']))
                        + '\n')


# def reacquire_api_results(none_id_entities):
#     counter = 0
#     for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]:
#         entity_list = none_id_entities[ent_type]
#         with open(REACQUIRED_API_RESULTS_PATH + ent_type + '.linked_wikiapi.jsonl', 'w+') as f:
#             for entity in tqdm(entity_list):
#                 tokenized_entity, tokenized_entity_without_the = wikidata_tools.tokenize_entity(entity)
#                 qent_1 = quote(tokenized_entity)
#                 qent_2 = quote(tokenized_entity_without_the)
#                 result = requests.get('https://en.wikipedia.org/w/api.php?action=query&format=json&prop=pageprops&titles='+qent_1+'|'+qent_2)
#                 f.write(json.dumps(dict(entity=entity, result=json.loads(result.text))) + '\n')
#                 counter += 1
#                 if counter == 3000:
#                     sleep(300)


def output_non_id_entities(none_id_entities, none_id_titles, all_sents):
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]:
        entity_list = none_id_entities[ent_type]
        linked_title_list = none_id_titles[ent_type]
        if not entity_list:
            continue
        with open(NONE_ID_ENTITY_PATH + ent_type + '.jsonl', 'w+') as f:
            for entity, linked_title in zip(entity_list, linked_title_list):
                f.write(json.dumps(dict(entity=entity, ent_text=" ".join(all_sents[entity[0]][entity[1]:entity[2]+1]),
                                        linked_title=linked_title)) + '\n')


def main():
    all_sents, all_tags = adver_tools.read_data(DATA_FILE_PATH)
    entities = load_linked_entities()
    ###
    # sample = {}
    # for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]:
    #     sample[ent_type] = entities[ent_type][:10]
    # acquire_via_api_output(sample, all_sents)
    ###
    acquire_via_api_output(entities, all_sents)
    api_results = read_api_results(API_RESULTS_PATH)
    entity_wikidata_dict, none_id_entities, none_id_titles = process_results(api_results)
    ###
    print("ENT_TYPE\tnum_of_entities_with_QID\tnum_of_entities_without_QID\tratio")
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]:
        print(ent_type + '\t' + str(len(entity_wikidata_dict[ent_type].keys()))
              + '\t' + str(len(none_id_entities[ent_type]))
              + '\t' + str(round(float(100*len(entity_wikidata_dict[ent_type].keys())
                                       / (len(entity_wikidata_dict[ent_type].keys())
                                          + len(none_id_entities[ent_type]))), 3)) + "%")
    # ###
    output_non_id_entities(none_id_entities, none_id_titles, all_sents)
    output_entity_wikidata_pair(entity_wikidata_dict, all_sents)


if __name__ == '__main__':
    main()
