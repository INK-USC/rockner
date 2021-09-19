import json




DATASET_NAME = "ontonotes_english"
DATASET_TYPE = "test"
ENTITY_TYPE_LIST_WITHOUT_PERSON = {"ontonotes_names": ['GPE', 'ORG', 'FAC', 'LOC', 'NORP', 'LAW', 'EVENT',
                                                       'WORK_OF_ART', 'PRODUCT', 'LANGUAGE'],
                                   "conll2003": ["ORG", "LOC"],
                                   "ontonotes_english": ['GPE', 'ORG', 'FAC', 'LOC', 'NORP', 'LAW', 'EVENT',
                                                         'WORK_OF_ART', 'PRODUCT', 'LANGUAGE']
                                   }

def load_class_instances():
    class_instances = {}
    counters = {}
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]:
        class_instances[ent_type] = {}
        counters[ent_type] = 0
        with open(DATASET_NAME + "/" + DATASET_TYPE + "/class_instances/" + ent_type + ".jsonl", "r") as f:
            for line in f:
                data = json.loads(line.strip())
                counters[ent_type] += len(data['instances'].items())
                class_instances[ent_type][data['class_id']] = {"class_title": data['class_title'],
                                                               "instances": data['instances']}
    return class_instances, counters


def process_class_intances(class_instances):
    processed_class_instances = {}
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]:
        processed_class_instances[ent_type] = {}
        for class_id, data in class_instances[ent_type].items():
            class_title = data['class_title']
            instances = data['instances']
            # processed_instances = {k: v for k, v in instances.items()
            #                        if k != v and all(ord(char) < 128 for char in v)}
            # if not processed_instances:
            #     print("No instances after processing", class_id, class_title)
            processed_class_instances[ent_type][class_id] = {"class_title": class_title,
                                                             "instance_num": len(instances),
                                                             "instances": instances}
    return processed_class_instances



def output_processed_class_instances(processed_class_instances):
    counters = {}
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]:
        counters[ent_type] = 0
        processed_class_instances[ent_type] = {k: v for k, v in sorted(processed_class_instances[ent_type].items(), key=lambda x: x[1]['instance_num'], reverse=True)}
        with open(DATASET_NAME + "/" + DATASET_TYPE + "/processed_class_instances/" + ent_type + ".jsonl", "w+") as f:
            for class_id, data in processed_class_instances[ent_type].items():
                counters[ent_type] += len(data['instances'].items())
                f.write(json.dumps(dict(class_id=class_id,
                                        class_title=data['class_title'],
                                        instance_num=data['instance_num'],
                                        instances=data['instances']))
                        + '\n')
    return counters


def main():
    class_instances, counters = load_class_instances()
    processed_class_instances = process_class_intances(class_instances)
    processed_counters = output_processed_class_instances(processed_class_instances)
    for ent_type in ENTITY_TYPE_LIST_WITHOUT_PERSON[DATASET_NAME]:
        print(ent_type + '\t\t\t' + str(counters[ent_type]) + '\t' + str(processed_counters[ent_type]))


if __name__ == '__main__':
    main()
