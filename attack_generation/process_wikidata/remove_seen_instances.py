import json
import wikidata_tools
import sys


sys.path.append("../../tools")
import adver_tools


DATASET_NAME = "ontonotes_english"
DATASET_TYPE = "train"
DATA_FILE_FOLDER = "../data/" + DATASET_NAME + "/"
FLAIR_PROCESSED_PATH = DATASET_NAME + "/" + DATASET_TYPE + "/flair_processed_class_instances/"
UNSEEN_FLAIR_PATH = DATASET_NAME + "/" + DATASET_TYPE + "/unseen_flair_processed_class_instances/"
ENTITY_TYPE_LIST_WITHOUT_PERSON = {"ontonotes_names": ['GPE', 'ORG', 'FAC', 'LOC', 'NORP', 'LAW', 'EVENT',
                                                       'WORK_OF_ART', 'PRODUCT', 'LANGUAGE'],
                                   "conll2003": ["ORG", "LOC"],
                                   "ontonotes_english": ['GPE', 'ORG', 'FAC', 'LOC', 'NORP', 'LAW', 'EVENT',
                                                         'WORK_OF_ART', 'PRODUCT', 'LANGUAGE']
                                   }

adver_tools.ENTITY_TYPE_LIST = adver_tools.ENTITY_TYPE_LIST_BY_DATASET_NAME[DATASET_NAME]
wikidata_tools.ENTITY_TYPE_LIST_WITHOUT_PERSON = ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]


def load_all_ents_surface_bound(path):
    all_ents = set()
    for dataset_type in ["train", "test", "dev"]:
        all_sents, all_tags = adver_tools.read_data(DATA_FILE_FOLDER + dataset_type + ".txt")
        entities_by_type, entities_by_sid, entities_set = adver_tools.update_NER_dict(all_sents, all_tags)
        all_ents.update({" ".join(k) for k in entities_set})
    return all_ents


def remove_seen_ents(all_ents, class_instances):
    unseen_class_instances = {}
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]:
        unseen_class_instances[ent_type] = {}
        for class_id, data in class_instances[ent_type].items():
            unseen_instances = {k: v for k, v in data['instances'].items() if not v.lower() in all_ents}
            unseen_class_instances[ent_type][class_id] = {"class_title": data['class_title'],
                                                          "instance_num": len(unseen_instances.keys()),
                                                          "instances": unseen_instances}
        unseen_class_instances[ent_type] = {k: v for k, v in sorted(unseen_class_instances[ent_type].items(),
                                                                    key=lambda x: x[1]['instance_num'],
                                                                    reverse=True)}
    return unseen_class_instances


def output_unseen_class_instances(path, unseen_class_instances):
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]:
        with open(path + ent_type + ".jsonl", "w+") as f:
            for class_id, data in unseen_class_instances[ent_type].items():
                f.write(json.dumps(dict(class_id=class_id,
                                        class_title=data['class_title'],
                                        instance_num=data['instance_num'],
                                        instances=data['instances'])) + '\n')


def main():
    all_ents_lower = load_all_ents_surface_bound(DATA_FILE_FOLDER)
    flair_processed_class_instances = wikidata_tools.load_class_instances(FLAIR_PROCESSED_PATH)
    unseen_class_instances = remove_seen_ents(all_ents_lower,flair_processed_class_instances)
    output_unseen_class_instances(UNSEEN_FLAIR_PATH, unseen_class_instances)
    person_instances = wikidata_tools.load_person_instances(FLAIR_PROCESSED_PATH)
    unseen_person_instances = {k: v for k, v in person_instances.items() if v.lower() not in all_ents_lower}
    with open(UNSEEN_FLAIR_PATH + "PERSON.jsonl", "w+") as f:
        for instance_id, instance_title in unseen_person_instances.items():
            f.write(json.dumps(dict(instance_id=instance_id, instance_title=instance_title)) + '\n')


if __name__ == '__main__':
    main()
