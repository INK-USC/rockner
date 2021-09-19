import json
import time
# import random
# import sys
import wikidata_tools
from wikidata2df import wikidata2df
from tqdm import tqdm


# sys.path.append("../attack_generation")
# import adver_tools


DATASET_NAME = "ontonotes_english"
DATASET_TYPE = "train"
CLASS_DATA_PATH = DATASET_NAME + "/" + DATASET_TYPE + "/class_data/"
CLASS_INSTANCE_PATH = DATASET_NAME + "/" + DATASET_TYPE + "/class_instances/"
NONE_INSTANCE_CLASS_PATH = DATASET_NAME + "/" + DATASET_TYPE + "/none_instance_classes/"
ENTITY_TYPE_LIST_WITHOUT_PERSON = {"ontonotes_names": ['GPE', 'ORG', 'FAC', 'LOC', 'NORP', 'LAW', 'EVENT',
                                                       'WORK_OF_ART', 'PRODUCT', 'LANGUAGE'],
                                   "conll2003": ["ORG", "LOC"],
                                   "ontonotes_english": ['GPE', 'ORG', 'FAC', 'LOC', 'NORP', 'LAW', 'EVENT',
                                                         'WORK_OF_ART', 'PRODUCT', 'LANGUAGE']
                                   }

wikidata_tools.ENTITY_TYPE_LIST_WITHOUT_PERSON = ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]


def query_instances_with_id(wikidata_ID):
    query = """
    SELECT ?item ?itemLabel
    WHERE
    {
        ?item wdt:P31 wd:%s.
        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    }
    LIMIT 1000
    """ % wikidata_ID
    dataframe = wikidata2df(query)
    return dataframe


def acquire_instances_via_api(class_ents_dict):
    queried_results = {}
    counter = 1
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]:
        class_instances = {}
        none_instance_class = {}
        class_ents = class_ents_dict[ent_type]
        for class_id, data in tqdm(class_ents.items()):
            if class_id in queried_results.keys():
                cached_results = queried_results[class_id]
                if not cached_results:
                    none_instance_class[class_id] = {"class_title": data['class_title'], "entities": data["entities"]}
                else:
                    class_instances[class_id] = {"class_title": data['class_title'],
                                                 "instance_num": len(cached_results.keys()),
                                                 "instances": cached_results}
                continue
            counter += 1
            time.sleep(0.1)
            if 0 == counter % 10:
                time.sleep(10)
                if 0 == counter % 60:
                    time.sleep(60)
            try:
                df = query_instances_with_id(class_id)
            except Exception as e:
                print(e)
                time.sleep(60)
                try:
                    df = query_instances_with_id(class_id)
                except Exception as e:
                    print(e)
                    none_instance_class[class_id] = {"class_title": data['class_title'], "entities": data["entities"]}
                    queried_results[class_id] = {}
                    continue
            raw_results = wikidata_tools.transfer_api_results(df)
            processed_results = {k: v for k, v in raw_results.items() if k != v and all(ord(char) < 128 for char in v)}
            if not processed_results:
                none_instance_class[class_id] = {"class_title": data['class_title'], "entities": data["entities"]}
                queried_results[class_id] = {}
                continue
            class_instances[class_id] = {"class_title": data['class_title'],
                                         "instance_num": len(processed_results.keys()),
                                         "instances": processed_results}
            queried_results[class_id] = processed_results
        output_class_instances(class_instances, ent_type)
        output_none_instance_classes(none_instance_class, ent_type)
    return None


def output_class_instances(class_instances, ent_type):
    class_instances = {k: v for k, v in sorted(class_instances.items(),
                                               key=lambda x: x[1]['instance_num'],
                                               reverse=True)}
    with open(CLASS_INSTANCE_PATH + ent_type + '.jsonl', 'w+') as f:
        for class_id, data in class_instances.items():
            f.write(json.dumps(dict(class_id=class_id,
                                    class_title=data['class_title'],
                                    instance_num=data['instance_num'],
                                    instances=data['instances']))
                    + '\n')


def output_none_instance_classes(none_instance_class, ent_type):
    if not none_instance_class:
        return None
    with open(NONE_INSTANCE_CLASS_PATH + ent_type + '.jsonl', 'w+') as f:
        for class_id, data in none_instance_class.items():
            f.write(json.dumps(dict(class_id=class_id,
                                    class_title=data['class_title'],
                                    entities=data['entities']))
                    + '\n')


def main():
    class_ents_dict = wikidata_tools.load_class_ents_dict(CLASS_DATA_PATH)

    ##
    # sample the first N entity_wikidata_pairs
    # sample_class_ents = {ent_type: {} for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]}
    # for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]:
    #     sample_class_ents[ent_type] = {k: class_ents_dict[ent_type][k] for k in list(class_ents_dict[ent_type].keys())[:3]}
    # acquire_instances_via_api(sample_class_ents)
    ##

    acquire_instances_via_api(class_ents_dict)


if __name__ == '__main__':
    main()
