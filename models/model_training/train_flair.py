from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from typing import List
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from pathlib import Path


def main():
    columns = {0: 'text', 1: 'ner'}

    data_folder = '../data/ontonotes_english_attacked'

    model_name = 'self_trained_flair_gpu_attacked'
    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                  train_file='train.txt',
                                  test_file='test.txt',
                                  dev_file='dev.txt')

    ###
    # downsampled_corpus = corpus.downsample(0.01)
    ###

    tag_type = 'ner'
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

    embedding_types: List[TokenEmbeddings] = [
        WordEmbeddings('en-crawl'),
        FlairEmbeddings('news-forward'),
        FlairEmbeddings('news-backward'),
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(
        embeddings=embedding_types)

    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type)

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    trainer.train('resources/taggers/' + model_name,
                  learning_rate=0.1,
                  mini_batch_size=64,
                  mini_batch_chunk_size=None,
                  max_epochs=100,
                  cycle_momentum=False,
                  anneal_factor=0.5,
                  patience=3,
                  initial_extra_patience=0,
                  min_learning_rate=0.0001,
                  train_with_dev=False,
                  monitor_train=False,
                  monitor_test=False,
                  embeddings_storage_mode="none",  # TODO: cpu
                  checkpoint=False,
                  save_final_model=True,
                  anneal_with_restarts=False,
                  anneal_with_prestarts=False,
                  batch_growth_annealing=False,
                  shuffle=True,
                  param_selection_mode=False,
                  write_weights=False,
                  num_workers=6,
                  sampler=None,
                  use_amp=False,  # TODO: True
                  amp_opt_level="O1",
                  eval_on_train_fraction=0.0,
                  eval_on_train_shuffle=False)


if __name__ == '__main__':
    main()
