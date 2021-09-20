# 🪨 RockNER: A Simple Method to Evaluate the Robustness of NER 
This is the Github repository of our paper, ***"RockNER: A Simple Method to Create Adversarial Examples for Evaluating the Robustness of Named Entity Recognition Models"*** (in Proc. of EMNLP2021).  

✍️  [***Bill Yuchen Lin***](https://yuchenlin.xyz/), [***Wenyang Gao***](), [***Jun Yan***](https://junyann.github.io/), [***Ryan Moreno***](https://ryan-moreno.github.io/), [***Xiang Ren***](http://www-bcf.usc.edu/~xiangren/) \
🏢  ***in Proceedings of EMNLP 2021 (short)*** \
🌐 Project website: [***https://inklab.usc.edu/rockner/***](https://inklab.usc.edu/rockner/).

 
## Paper Abstract 
To audit the robustness of named entity recognition (NER) models, we propose RockNER, a simple yet effective method to create natural adversarial examples. Specifically, at the entity level, we replace target entities with other entities of the same semantic class in Wikidata; at the context level, we use pre-trained language models (e.g., BERT) to generate word substitutions. Together, the two levels of at- tack produce natural adversarial examples that result in a shifted distribution from the training data on which our target models have been trained. We apply the proposed method to the OntoNotes dataset and create a new benchmark named OntoRock for evaluating the robustness of existing NER models via a systematic evaluation protocol. Our experiments and analysis reveal that even the best model has a significant performance drop, and these models seem to memorize in-domain entity patterns instead of reasoning from the context. Our work also studies the effects of a few simple data augmentation methods to improve the robustness of NER models.
<!-- \footnote{Our code and data are publicly available at the project website: \url{https://inklab.usc.edu/rockner}.} -->

![intro](https://inklab.usc.edu/rockner/images/introduction.png){: style="border: 0px solid black"}

## Resources 
Please download our ***OntoRock*** dataset by filling the [***form***](https://forms.gle/ydhayvV1uFGdLkch9) here and the link will show up once you read the disclaimer and submit it. There are eight files as follows:


### Dataset Format

1. `Original-OntoNotes_train.txt` (1,148,427 lines)
    - The original training data of OntoNotes.
1. `OntoRock-Full_dev.txt` (161,123 lines)
    - The development data of OntoRock-Full.
1. `OntoRock-Entity_dev.txt` (161,152 lines)
    - The development data of OntoRock-Entity.
1. `OntoRock-Context_dev.txt` (156,215 lines)
    - The development data of OntoRock-Context.
1. `Original-OntoNotes_test_pub.txt` (160,989 lines)
    - The original test data of OntoNotes, where the truth tags are hidden.
1. `OntoRock-Full_test_pub.txt` (165,872 lines)
    - The test data of OntoRock-Full, where the truth tags are hidden.
1. `OntoRock-Entity_test_pub.txt` (165,906 lines)
    - The test data of OntoRock-Entity, where the truth tags are hidden.
1. `OntoRock-Context_test_pub.txt` (160,953 lines)
    - The test data of OntoRock-Context, where the truth tags are hidden.

```
# a sentence in our txt file, truth tags are hidden in test files
We O
respectfully O
invite O
you O
to O
watch O
a O
special O
edition O
of O
Across B-ORG
China I-ORG
. O
# sentences are separated by blank line
```
 
## Contact
This repo is now under active development, and there may be issues caused by refactoring code.
Please email ***yuchen.lin@usc.edu*** if you have any questions.

## Citation

```bibtex
@inproceedings{lin-etal-2021-rockner,
    title = "RockNER: A Simple Method to Create Adversarial Examples for Evaluating the Robustness of Named Entity Recognition Models",
    author = "Lin, Bill Yuchen and Gao, Wenyang and Yan, Jun and Moreno, Ryan and Ren, Xiang",
    booktitle = "Proc. of EMNLP (short paper)",
    year = "2021",
    note={to appear}
}
```