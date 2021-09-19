import wikidata_tools


DATASET_NAME = "ontonotes_english"
DATASET_TYPE = "dev"
ENTITY_WIKIDATA_PATH = DATASET_NAME + "/" + DATASET_TYPE + "/entity_wikidata_pair/"
CLASS_INSTANCES_PATH = DATASET_NAME + "/" + DATASET_TYPE + "/class_instances/"
FLAIR_PROCESSED_INSTANCES_PATH = DATASET_NAME + "/" + DATASET_TYPE + "/flair_processed_class_instances/"
UNSEEN_PROCESSED_CLASS_INSTANCES_PATH = DATASET_NAME + "/" + DATASET_TYPE + "/unseen_flair_processed_class_instances/"
ENTITY_TYPE_LIST_WITHOUT_PERSON = {"ontonotes_names": ['GPE', 'ORG', 'FAC', 'LOC', 'NORP', 'LAW', 'EVENT',
                                                       'WORK_OF_ART', 'PRODUCT', 'LANGUAGE'],
                                   "conll2003": ["ORG", "LOC"],
                                   "ontonotes_english": ['GPE', 'ORG', 'FAC', 'LOC', 'NORP', 'LAW', 'EVENT',
                                                         'WORK_OF_ART', 'PRODUCT', 'LANGUAGE']
                                   }


wikidata_tools.ENTITY_TYPE_LIST_WITHOUT_PERSON = ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]


def count_unique_QID(entity_wikidata_pair):
    unique_QID_set = {}
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]:
        unique_QID_set[ent_type] = set()
        for entity, data in entity_wikidata_pair[ent_type].items():
            unique_QID_set[ent_type].add(data['wikidata']['wikidata_ID'])
    unique_QID_counters = {k: len(v) for k, v in unique_QID_set.items()}
    return unique_QID_counters


def count_instance_num(class_instances):
    instance_counters = {}
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]:
        instance_counters[ent_type] = 0
        for class_id, data in class_instances[ent_type].items():
            instance_counters[ent_type] += data['instance_num']
    return instance_counters


def main():
    entity_wikidata_dict = wikidata_tools.load_entity_wikidata_pair(ENTITY_WIKIDATA_PATH)
    class_instances = wikidata_tools.load_class_instances(CLASS_INSTANCES_PATH)
    flair_processed_class_instances = wikidata_tools.load_class_instances(FLAIR_PROCESSED_INSTANCES_PATH)
    unseen_class_instances = wikidata_tools.load_class_instances(UNSEEN_PROCESSED_CLASS_INSTANCES_PATH)
    unique_QID_counters = count_unique_QID(entity_wikidata_dict)
    instance_counters = count_instance_num(class_instances)
    flair_instance_counters = count_instance_num(flair_processed_class_instances)
    unseen_instances_counters = count_instance_num(unseen_class_instances)

    print("ENT_TYPE,QID_NUM,INSTANCE_NUM,FLAIR_INSTANCE_NUM")
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]:
        print(ent_type + ',' + str(unique_QID_counters[ent_type]) + ',' + str(instance_counters[ent_type])
              + ',' + str(flair_instance_counters[ent_type]) + ',' + str(unseen_instances_counters[ent_type]))


if __name__ == '__main__':
    main()
