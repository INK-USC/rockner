import json
import time
import wikidata_tools
import sys
from wikidata2df import wikidata2df
from tqdm import tqdm


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

wikidata_tools.ENTITY_TYPE_LIST_WITHOUT_PERSON = ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]


def query_classes_with_id(wikidata_ID):
    query = """
    SELECT ?item ?itemLabel
    WHERE
    {
        wd:%s wdt:P31 ?item.
        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    }
    """ % wikidata_ID
    dataframe = wikidata2df(query)
    return dataframe


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


def load_wikidata():
    entity_wikidata_dict = {}
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]:
        entity_wikidata_dict[ent_type] = {}
        with open(DATASET_NAME + '/' + DATASET_TYPE + '/entity_wikidata_pair/'
                  + ent_type + '.linked_wikidata.jsonl', 'r') as f:
            for line in f:
                data = dict(json.loads(line.strip()))
                entity_wikidata_dict[ent_type][tuple(data['entity'])] = data['wikidata']
    return entity_wikidata_dict


def output_all_class_entities(all_class_entities_dict):
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]:
        with open(DATASET_NAME + '/' + DATASET_TYPE + '/class_data/'
                  + ent_type + '.class_ents.jsonl', 'w+') as f:
            for q_id, data in all_class_entities_dict[ent_type].items():
                f.write(json.dumps(dict(class_id=q_id,
                                        class_title=data['class_title'],
                                        entities=data['entities'])) + '\n')


def output_all_entity_classes(all_entity_classes):
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]:
        with open(DATASET_NAME + '/' + DATASET_TYPE + '/class_data/'
                  + ent_type + '.ent_classes.jsonl', 'w+') as f:
            for entity, data in all_entity_classes[ent_type].items():
                f.write(json.dumps(dict(entity=entity, ent_id=data['ent_id'], classes=data['classes'])) + '\n')


def output_all_none_class_entities(all_none_class_entities):
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]:
        if not all_none_class_entities[ent_type]:
            continue
        with open(DATASET_NAME + '/' + DATASET_TYPE + '/none_class_entities/'
                  + ent_type + '.jsonl', 'w+') as f:
            for entity, ent_id in all_none_class_entities[ent_type].items():
                f.write(json.dumps(dict(entity=entity, ent_id=ent_id)) + '\n')


def output_class_entities(class_entities_dict, ent_type):
    processed_class_ents_dict = {k: {"class_title": v['class_title'],
                                     "ent_num": sum([len(ent_v) for ent_v in v['entities'].values()]),
                                     "QID_num": len(v['entities'].keys()),
                                     "entities": v['entities']}
                                 for k, v in class_entities_dict.items()}
    processed_class_ents_dict = {k: v for k, v in sorted(processed_class_ents_dict.items(),
                                                         key=lambda x: x[1]['ent_num'],
                                                         reverse=True)}
    with open(DATASET_NAME + '/' + DATASET_TYPE + '/class_data/'
              + ent_type + '.class_ents.jsonl', 'w+') as f:
        for q_id, data in processed_class_ents_dict.items():
            entities = data['entities']
            f.write(json.dumps(dict(class_id=q_id,
                                    class_title=data['class_title'],
                                    ent_num=data['ent_num'],
                                    QID_num=data['QID_num'],
                                    entities=entities)) + '\n')


def output_entity_classes(entity_classes, ent_type):
    with open(DATASET_NAME + '/' + DATASET_TYPE + '/class_data/'
              + ent_type + '.ent_classes.jsonl', 'w+') as f:
        for entity, data in entity_classes.items():
            f.write(json.dumps(dict(entity=entity, 
                                    ent_text=data['ent_text'], 
                                    linked_title=data['linked_title'],
                                    ent_id=data['ent_id'],
                                    class_num=len(data['classes'].keys()),
                                    classes=data['classes'])) 
                    + '\n')


def output_none_class_entities(none_class_entities, ent_type):
    if not none_class_entities:
        return None
    with open(DATASET_NAME + '/' + DATASET_TYPE + '/none_class_entities/'
              + ent_type + '.jsonl', 'w+') as f:
        for entity, data in none_class_entities.items():
            f.write(json.dumps(dict(entity=entity,
                                    ent_text=data['ent_text'],
                                    linked_title=data['linked_title'],
                                    ent_id=data['ent_id'])) 
                    + '\n')


def acquire_classes(entity_wikidata, all_sents):
    counter = 1
    all_class_entities_dict = {}
    all_entity_classes_dict = {}
    queried_entity_results = {}
    all_none_class_entities = {}
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]:
        class_entities_dict = {}
        entity_classes_dict = {}
        none_class_entities = {}
        for entity, wikidata in tqdm(entity_wikidata[ent_type].items()):
            entity_id = wikidata['wikidata_ID']
            entity_text = " ".join(all_sents[entity[0]][entity[1]:entity[2]+1])
            linked_title = wikidata['title']
            if entity_id in queried_entity_results.keys():
                classes_results = queried_entity_results[entity_id]
                if not classes_results:
                    none_class_entities[tuple(entity)] = {"ent_text": entity_text, 
                                                          "ent_id": entity_id, 
                                                          "linked_title": linked_title}
                    continue
                entity_classes_dict[tuple(entity)] = {"ent_text": entity_text, "linked_title": linked_title, 
                                                      "ent_id": entity_id, "classes": classes_results}
                for class_id, class_title in classes_results.items():
                    if class_id in class_entities_dict.keys():
                        if entity_id in class_entities_dict[class_id]["entities"].keys():
                            class_entities_dict[class_id]["entities"][entity_id].append(dict(entity=entity,
                                                                                             ent_text=entity_text,
                                                                                             linked_title=linked_title))
                        else:
                            class_entities_dict[class_id]["entities"][entity_id] = [dict(entity=entity,
                                                                                         ent_text=entity_text,
                                                                                         linked_title=linked_title)]
                    else:
                        class_entities_dict[class_id] = {"class_title": class_title,
                                                         "entities": {entity_id: [dict(entity=entity,
                                                                                       ent_text=entity_text,
                                                                                       linked_title=linked_title)]}}
            else:
                try:
                    df = query_classes_with_id(entity_id)
                except:
                    print('HTTP ERROR 429')
                    time.sleep(60)
                    try:
                        df = query_classes_with_id(entity_id)
                    except:
                        print('HTTP ERROR 429 AGAIN')
                        time.sleep(300)
                        try:
                            df = query_classes_with_id(entity_id)
                        except:
                            none_class_entities[tuple(entity)] = {"ent_text": entity_text, 
                                                                  "ent_id": entity_id, 
                                                                  "linked_title": linked_title}
                classes_results = wikidata_tools.transfer_api_results(df)
                if not classes_results:
                    none_class_entities[tuple(entity)] = {"ent_text": entity_text, 
                                                          "ent_id": entity_id, 
                                                          "linked_title": linked_title}
                    continue
                entity_classes_dict[tuple(entity)] = {"ent_text": entity_text, "linked_title": linked_title, 
                                                      "ent_id": entity_id, "classes": classes_results}
                queried_entity_results[entity_id] = classes_results
                for class_id, class_title in classes_results.items():
                    if class_id in class_entities_dict.keys():
                        if entity_id in class_entities_dict[class_id]["entities"].keys():
                            class_entities_dict[class_id]["entities"][entity_id].append(dict(entity=entity,
                                                                                             ent_text=entity_text,
                                                                                             linked_title=linked_title))
                        else:
                            class_entities_dict[class_id]["entities"][entity_id] = [dict(entity=entity,
                                                                                         ent_text=entity_text,
                                                                                         linked_title=linked_title)]
                    else:
                        class_entities_dict[class_id] = {"class_title": class_title,
                                                         "entities": {entity_id: [dict(entity=entity,
                                                                                       ent_text=entity_text,
                                                                                       linked_title=linked_title)]}}
                counter += 1
                time.sleep(0.1)
                if 0 == counter % 10:
                    time.sleep(10)
                    if 0 == counter % 60:
                        time.sleep(60)
        all_class_entities_dict[ent_type] = class_entities_dict
        all_entity_classes_dict[ent_type] = entity_classes_dict
        all_none_class_entities[ent_type] = none_class_entities
        output_class_entities(class_entities_dict, ent_type)
        output_entity_classes(entity_classes_dict, ent_type)
        output_none_class_entities(none_class_entities, ent_type)
    return all_class_entities_dict, all_entity_classes_dict, all_none_class_entities


def main():
    all_sents, all_tags = adver_tools.read_data(DATA_FILE_PATH)
    entity_wikidata = load_wikidata()

    ##
    # sample the first N entity_wikidata_pairs
    # sample_ent_wiki = {ent_type: {} for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]}
    # for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]:
    #     sample_ent_wiki[ent_type] = {k: entity_wikidata[ent_type][k] for k in list(entity_wikidata[ent_type].keys())[:2]}
    # all_class_entities_dict, all_entity_classes_dict, all_none_class_entities = acquire_classes(sample_ent_wiki, all_sents)
    ##

    all_class_entities_dict, all_entity_classes_dict, all_none_class_entities = acquire_classes(entity_wikidata, all_sents)
    
    ###
    print("ENT_TYPE,NUM_HAS_CLASS,NUM_HAS_NONE_CLASS,RATIO")
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]:
        num_has_class = len(all_entity_classes_dict[ent_type].keys())
        num_has_none_class = len(all_none_class_entities[ent_type].keys())
        ratio = str(round(float(num_has_class / (num_has_class + num_has_none_class) * 100), 3)) + '%'
        print(ent_type + ',' + str(num_has_class) + ',' + str(num_has_none_class) + ',' + ratio)
    ###


if __name__ == "__main__":
    main()
