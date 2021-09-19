import os

DATASET_NAME = "ontonotes_english"
DATASET_TYPE = "dev"
MODEL = "spacy_lg"
ARGS_ENTITY_LEVEL_BASIC = dict(attack_num=5, seeds=[1, 2, 3, 4, 5])
ARGS = ARGS_ENTITY_LEVEL_BASIC

MODEL_FILE_FOLDER_NAME = MODEL + "_entity_" + str(ARGS['attack_num']) + "attacks_concurrent_unseen_deleted"
OUTPUT_MAIN_PATH = "../../rockner/" + DATASET_NAME + "/" + DATASET_TYPE + "/" + MODEL_FILE_FOLDER_NAME + "/"
F1_REPORTS_PATH = OUTPUT_MAIN_PATH + "f1_reports/"
RESULTS_PATH = OUTPUT_MAIN_PATH + "results/"

for attack_round in range(ARGS['attack_num']):
    os.system("python conlleval.py <" + RESULTS_PATH + str(attack_round) + ".results > " + F1_REPORTS_PATH + str(attack_round) + ".f1_report.txt")
