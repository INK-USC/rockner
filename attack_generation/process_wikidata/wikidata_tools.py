import nltk
import json
import re
from nltk.corpus import stopwords

stopwords_en = stopwords.words('english')
ENTITY_TYPE_LIST_WITHOUT_PERSON = ['GPE', 'ORG', 'FAC', 'LOC', 'NORP', 'LAW', 'EVENT', 'WORK_OF_ART', 'PRODUCT',
                                   'LANGUAGE']


def get_dict_value(in_dict, target_key, results=None, not_d=True):
    if results is None:
        results = []
    for key in in_dict.keys():
        data = in_dict[key]
        if isinstance(data, dict):
            get_dict_value(data, target_key, results=results, not_d=not_d)
        if key == target_key and isinstance(data, dict) != not_d:
            results.append(in_dict[key])
    return results


def get_wikidata_from_api_result(pages):
    data = {}
    for key, value in pages.items():
        wikidata_ID = get_dict_value(value, 'wikibase_item')
        if wikidata_ID:
            desc = get_dict_value(value, 'wikibase-shortdesc')
            if desc and re.search('Disambiguation', desc[0]):
                continue
            data['pageid'] = key
            data['title'] = value['title']
            data['wikidata_ID'] = wikidata_ID[0]
    return data


def tokenize_entity(entity):
    entity_words = entity.split()
    tmp_cap_entity_words = [ent.title() for ent in entity_words]
    cap_entity_words = tmp_cap_entity_words.copy()
    for index, word in enumerate(tmp_cap_entity_words):
        if index == 0:
            cap_entity_words[index] = word.title()
            continue
        word_l = word.lower()
        if word == ' - ':
            cap_entity_words[index] = '-'
        elif word_l == 'versus':
            cap_entity_words[index] = 'v.'
        elif word_l in stopwords_en:
            cap_entity_words[index] = word.lower()
    processed_entity = " ".join(cap_entity_words)
    processed_entity_without_the = " ".join(ent for ent in processed_entity.split() if ent != 'The')
    return processed_entity, processed_entity_without_the


def transfer_api_results(dataframe):
    values = {}
    for index, row in dataframe.iterrows():
        item_wikidata_ID = row['item']
        item_title = row['itemLabel']
        values[item_wikidata_ID] = item_title
    return values


def load_instances(path):
    all_instances = {k: {} for k in ENTITY_TYPE_LIST_WITHOUT_PERSON}
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON:
        with open(path + ent_type + '.all_instances.jsonl', 'r') as f:
            for line in f:
                data = dict(json.loads(line.strip()))
                all_instances[ent_type][data['instance_id']] = data['instance_data']
    return all_instances


def load_instances_as_set(path):
    all_instances = {k: set() for k in ENTITY_TYPE_LIST_WITHOUT_PERSON}
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON:
        with open(path + ent_type + '.all_instances.jsonl', 'r') as f:
            for line in f:
                data = dict(json.loads(line.strip()))
                all_instances[ent_type].add(data['instance_data']['title'])
    return all_instances


def load_instance_joint_data(path):
    all_instances = {k: {} for k in ENTITY_TYPE_LIST_WITHOUT_PERSON}
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON:
        with open(path + ent_type + '.all_instances.jsonl', 'r') as f:
            for line in f:
                data = dict(json.loads(line.strip()))
                all_instances[ent_type][data['instance_id']] = {'instance_data': data['instance_data'],
                                                                'derive_from': data['derive_from']}
    return all_instances


def load_instance_dict(path):
    all_instances = {k: {} for k in ENTITY_TYPE_LIST_WITHOUT_PERSON}
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON:
        with open(path + ent_type + '.all_instances.jsonl', 'r') as f:
            for line in f:
                data = dict(json.loads(line.strip()))
                if not all_instances[ent_type].get(data['instance_data']['title']):
                    all_instances[ent_type][data['instance_data']['title']] = data['derive_from']
                else:
                    all_instances[ent_type][data['instance_data']['title']].update(data['derive_from'])
    return all_instances


def output_instances(path, instances):
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON:
        with open(path + ent_type + '.all_instances.jsonl', 'w+') as f:
            for item_id, item_data in instances[ent_type].items():
                f.write(json.dumps(dict(instance_id=item_id, instance_data=item_data)) + '\n')


def load_instance_class( path ):
    instance_class = {k: {} for k in ENTITY_TYPE_LIST_WITHOUT_PERSON}
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON:
        with open(path + ent_type + '.class_instances.jsonl', 'r') as f:
            for line in f:
                data = dict(json.loads(line.strip()))
                class_id = data['class_id']
                instances = data['instances']
                for instance_id, title in instances.items():
                    if not instance_class[ent_type].get(instance_id):
                        instance_class[ent_type][instance_id] = {class_id}
                    else:
                        instance_class[ent_type][instance_id].add(class_id)
    return instance_class


def load_ext_class_superclass(path):
    ext_class_superclass = {k: {} for k in ENTITY_TYPE_LIST_WITHOUT_PERSON}
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON:
        with open(path + ent_type + '.superclass_extended_classes.jsonl', 'r') as f:
            for line in f:
                data = dict(json.loads(line.strip()))
                superclass = {'id': data['superclass_id'], 'title': data['superclass_title']}
                ext_classes = data['extended_classes']
                for ext_class_id, title in ext_classes.items():
                    if not ext_class_superclass[ent_type].get(ext_class_id):
                        ext_class_superclass[ent_type][ext_class_id] = {'title': title,
                                                                        'superclasses': {superclass['id']: superclass['title']}}
                    else:
                        ext_class_superclass[ent_type][ext_class_id]['superclasses'][superclass['id']] = superclass['title']
    return ext_class_superclass


def load_superclass_src_class(path):
    superclass_src_class = {k: {} for k in ENTITY_TYPE_LIST_WITHOUT_PERSON}
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON:
        with open(path + ent_type + '.class_superclasses.jsonl', 'r') as f:
            for line in f:
                data = dict(json.loads(line.strip()))
                src_class = {'id': data['class_id'], 'title': data['class_title']}
                superclasses = data['superclasses']
                for superclass_id, title in superclasses.items():
                    if not superclass_src_class[ent_type].get(superclass_id):
                        superclass_src_class[ent_type][superclass_id] = {'title': title,
                                                                         'src_classes': {src_class['id']: src_class['title']}}
                    else:
                        superclass_src_class[ent_type][superclass_id]['src_classes'][src_class['id']] = src_class['title']
    return superclass_src_class


def load_person_instances(path):
    all_instances = {}
    with open(path + "PERSON" + '.jsonl', 'r') as f:
        for line in f:
            data = dict(json.loads(line.strip()))
            all_instances[data['instance_id']] = data['instance_title']
    return all_instances


def load_person_instances_as_set(path):
    all_instances = set()
    with open(path + 'PERSON' + '.jsonl', 'r') as f:
        for line in f:
            data = dict(json.loads(line.strip()))
            all_instances.add(data['instance_title'])
    return all_instances


def load_entity_wikidata_pair(path):
    entity_wikidata_dict = {}
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON:
        entity_wikidata_dict[ent_type] = {}
        with open(path + ent_type + '.linked_wikidata.jsonl', 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                ent = data['entity']
                entity_wikidata_dict[ent_type][tuple(ent)] = {"ent_text": data['ent_text'],
                                                              "linked_title": data['linked_title'],
                                                              "wikidata": data['wikidata']}
    return entity_wikidata_dict


def load_class_ents_dict(path):
    class_ents_dict = {ent_type: {} for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON}
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON:
        with open(path + ent_type + ".class_ents.jsonl", 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                class_ents_dict[ent_type][data['class_id']] = dict(class_title=data['class_title'],
                                                                   ent_num=data['ent_num'],
                                                                   QID_num=data['QID_num'],
                                                                   entities=data['entities'])
    return class_ents_dict


def load_ent_classes_dict(path):
    ent_classes = {ent_type: {} for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON}
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON:
        with open(path + ent_type + ".ent_classes.jsonl", "r") as f:
            for line in f:
                data = json.loads(line.strip())
                ent_classes[ent_type][tuple(data['entity'])] = {"ent_text": data['ent_text'],
                                                                "linked_title": data['linked_title'],
                                                                "ent_id": data['ent_id'],
                                                                "classes": data['classes']}
    return ent_classes


def load_class_instances(path):
    class_instances = {}
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON:
        class_instances[ent_type] = {}
        with open(path + ent_type + ".jsonl", "r") as f:
            for line in f:
                data = json.loads(line.strip())
                class_instances[ent_type][data['class_id']] = {"class_title": data['class_title'],
                                                               "instance_num": data['instance_num'],
                                                               "instances": data['instances']}
    return class_instances

# def load_superclass_class(path):
#     all_superclasses = {k: {} for k in ENTITY_TYPE_LIST_WITHOUT_PERSON}
#     for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON:
#         with open(path + ent_type + '.class_superclasses.jsonl', 'r') as f:
#             for line in f:
#                 data = dict(json.loads(line.strip()))
#                 all_superclasses[ent_type][data['superclass']]
