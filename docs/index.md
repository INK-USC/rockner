---
layout: default
title: RockNER
nav_order: 1
description: "RockNER: A Simple Method to Create Adversarial Examples for Evaluating the Robustness of Named Entity Recognition Models | EMNLP 2021"
permalink: /
last_modified_date: Jun 5th 2021
toc_list: true
---


<!-- <link href="http://allfont.net/allfont.css?fonts=agency-fb-bold" rel="stylesheet" type="text/css" /> -->

<style>
@font-face{font-family:agency fb bold;font-style:normal;font-weight:700;src:local('Agency FB Bold'),local('AgencyFB-Bold'),url(http://allfont.net/cache/fonts/agency-fb-bold_cee84847c4ab16cf2b0952d063712724.woff) format('woff'),url(http://allfont.net/cache/fonts/agency-fb-bold_cee84847c4ab16cf2b0952d063712724.ttf) format('truetype')}

p, li{
    font-size: 16px;
} 



.acc{
    font-weight: 700;
    color: green;
    text-align: center;
}

.modelname{
    font-weight: 650;
    text-align: center;
    color: blue;
}

.submitter{
    font-weight: 500;
    text-align: center;
    color: purple;
}

.date{
    font-weight: 500;
    text-align: center;
}

.traindata{
    font-weight: 600;
    text-align: center;
    color: purple;
}
/* #main-content {
    float: center;
    width: auto; } */
</style>


# 🪨 RockNER: A Simple Method to Create Adversarial Examples for Evaluating the Robustness of NER Models
{: .no_toc style="font-weight: 600; font-family: 'Agency FB Bold', arial;"}
✍️  [***Bill Yuchen Lin***](https://yuchenlin.xyz/){: target="blank"}, [***Wenyang Gao***](){: target="blank"}, [***Jun Yan***](https://junyann.github.io/), [***Ryan Moreno***](https://ryan-moreno.github.io/){: target="blank"}, [***Xiang Ren***](http://www-bcf.usc.edu/~xiangren/){: target="blank"} \
🏢  ***in Proceedings of EMNLP 2021 (short)***


![intro](images/authors.png){: style="border: 2px solid black; width: 90%; center"}

--- 
 

## Quick Links
{: .no_toc}
<!-- {: .fs-7 .fw-700 .text-blue-300 } -->
<span class="fs-4">
[Paper](paper.pdf){: target="_blank" .btn .btn-green .mr-1 .fs-4}
<!-- [Video](https://s3.amazonaws.com/pf-upload-01/u-59356/0/2021-06-25/8f53jb9/riddlesense_acl.mp4){: target="_blank" .btn .btn-green .mr-1 .fs-4} -->
[Github](https://github.com/INK-USC/RockNER/){: target="_blank" .btn .btn-purple .mr-1 .fs-4 }
[Dataset](https://forms.gle/iWdsgN44TeoXW19e6){: target="_blank" .btn .btn-blue .mr-1 .fs-4 }
[Leaderboard](#leaderboard){: .btn .btn-red .mr-1 .fs-4 }
<!-- [Download MickeyCorpus](https://forms.gle/fCxN1YAyqKpQ4cXNA){: target="_blank" .btn .btn-blue .mr-1 .fs-3 }
[Download X-CSR Datasets](https://forms.gle/gVCNgVXr1tyYkDya9){: target="_blank" .btn .btn-blue .mr-1 .fs-3 } -->
<!-- [Video](https://mega.nz/file/5SpQjJKS#J82pfZVDzy3r4aWdNF4R6O8EP5gsepbY20vYihANfgE){: target="_blank" .btn .btn-blue .mr-1 .fs-3 }
[Slides](/opencsr_naacl_slides.pptx){: target="_blank" .btn .btn-red .mr-1 .fs-3 } -->
</span> 
<!-- 
[***Intro***](#intro){: .mr-1 .fs-5} 
[***Leaderboard***](#leaderboard){:  .mr-1 .fs-5 } 
[***Citation***](#citation){: mr-1 .fs-5 } -->
<!-- - TOC
{:toc} -->

<!-- [Download MickeyCorpus](https://forms.gle/fCxN1YAyqKpQ4cXNA){: target="_blank" .btn .btn-blue .mr-1 .fs-3 }
[Download X-CSR Datasets](https://forms.gle/gVCNgVXr1tyYkDya9){: target="_blank" .btn .btn-blue .mr-1 .fs-3 } -->
<!-- [Video](https://mega.nz/file/5SpQjJKS#J82pfZVDzy3r4aWdNF4R6O8EP5gsepbY20vYihANfgE){: target="_blank" .btn .btn-blue .mr-1 .fs-3 }
[Slides](/opencsr_naacl_slides.pptx){: target="_blank" .btn .btn-red .mr-1 .fs-3 } -->
---
 
## Intro
<!-- This is the project site for the paper, [_Differentiable Cross-Lingual Commonsense Reasoning_](https://www.aclweb.org/anthology/2021.naacl-main.366/){: target="_blank"}, by [_Bill Yuchen Lin_](https://yuchenlin.xyz/){: target="_blank"}, [_Haitian Sun_](https://scholar.google.com/citations?user=opSHsTQAAAAJ&hl=en){: target="_blank"}, [_Bhuwan Dhingra_](http://www.cs.cmu.edu/~bdhingra/){: target="_blank"}, [_Manzil Zaheer_](https://scholar.google.com/citations?user=A33FhJMAAAAJ&hl=en){: target="_blank"}, [_Xiang Ren_](http://ink-ron.usc.edu/xiangren/){: target="_blank"}, and [_William W. Cohen_](https://wwcohen.github.io/){: target="_blank"}, in Proc. of [*NAACL 2021*](https://2021.naacl.org/){: target="_blank"}. 
This is a joint work by Google Research and USC. -->

 

<!-- ##  --> 
<!-- ***Abstract.***{: .text-red-100}  -->
 
To audit the robustness of named entity recognition (NER) models, we propose RockNER, a simple yet effective method to create natural adversarial examples. Specifically, at the entity level, we replace target entities with other entities of the same semantic class in Wikidata; at the context level, we use pre-trained language models (e.g., BERT) to generate word substitutions. Together, the two levels of at- tack produce natural adversarial examples that result in a shifted distribution from the training data on which our target models have been trained. We apply the proposed method to the OntoNotes dataset and create a new benchmark named OntoRock for evaluating the robustness of existing NER models via a systematic evaluation protocol. Our experiments and analysis reveal that even the best model has a significant performance drop, and these models seem to memorize in-domain entity patterns instead of reasoning from the context. Our work also studies the effects of a few simple data augmentation methods to improve the robustness of NER models.
<!-- \footnote{Our code and data are publicly available at the project website: \url{https://inklab.usc.edu/rockner}.} -->

![intro](images/introduction.png){: style="border: 0px solid black"}

## Dataset Format

Please download our dataset by filling the [***form***](https://forms.gle/iWdsgN44TeoXW19e6){: target="_blank"} here and the link will show up once you read the disclaimer and submit it. There are eight files as follows:

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


## Leaderboard

<table>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th rowspan="2">Submitter</th>
    <th rowspan="2">Ori_test_F1</th>
    <th colspan="3">OntoRock_test_F1</th>
  </tr>
  <tr>
    <th>Ent</th>
    <th>Cont</th>
    <th>Full</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="modelname"><a href="https://arxiv.org/abs/1907.11692" target="_blank">RoBERTa-CRF </a></td>
    <td class="submitter"><a href="http://inklab.usc.edu" target="_blank" title="9/10/2021">USC-INK</a></td>
    <td class="traindata">92.4</td>
    <td class="acc">63.4</td>
    <td class="acc">87.2</td>
    <td class="acc">58.5</td>
  </tr>
  <tr>
    <td class="modelname"><a href="https://aclanthology.org/C18-1139/" target="_blank">Flair </a></td>
    <td class="submitter"><a href="http://inklab.usc.edu" target="_blank" title="9/10/2021">USC-INK</a></td>
    <td class="traindata">90.7</td>
    <td class="acc">59.6</td>
    <td class="acc">86.1</td>
    <td class="acc">55.3</td>
  </tr>
  <tr>
    <td class="modelname"><a href="https://aclanthology.org/N19-1423/" target="_blank">BERT-CRF </a></td>
    <td class="submitter"><a href="http://inklab.usc.edu" target="_blank" title="9/10/2021">USC-INK</a></td>
    <td class="traindata">90.6</td>
    <td class="acc">59.2</td>
    <td class="acc">85.8</td>
    <td class="acc">54.6</td>
  </tr>
  <tr>
    <td class="modelname"><a href="https://aclanthology.org/2020.acl-demos.14/" target="_blank">Stanza </a></td>
    <td class="submitter"><a href="http://inklab.usc.edu" target="_blank" title="9/10/2021">USC-INK</a></td>
    <td class="traindata">87.9</td>
    <td class="acc">56.1</td>
    <td class="acc">83.0</td>
    <td class="acc">51.7</td>
  </tr>
  <tr>
    <td class="modelname"><a href="https://zenodo.org/record/5226955#.YT4qx9MzbK0" target="_blank">Spacy </a></td>
    <td class="submitter"><a href="http://inklab.usc.edu" target="_blank" title="9/10/2021">USC-INK</a></td>
    <td class="traindata">87.3</td>
    <td class="acc">43.9</td>
    <td class="acc">81.8</td>
    <td class="acc">40.1</td>
  </tr>
  <tr>
    <td class="modelname"><a href="https://aclanthology.org/N16-1030/" target="_blank">BLSTM-CRF </a></td>
    <td class="submitter"><a href="http://inklab.usc.edu" target="_blank" title="9/10/2021">USC-INK</a></td>
    <td class="traindata">84.6</td>
    <td class="acc">40.5</td>
    <td class="acc">77.3</td>
    <td class="acc">32.4</td>
  </tr>
</tbody>
</table>

### Submission Guide (Todo)
{: .no_toc}

This is [***an example submission file***](submission_example.jsonl){: target="_blank"}. Please submit your prediction file and information via [***this form***](https://forms.gle/a3yyoxmgj1FoJpMM7){: target="_blank"}.

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
{: .fs-4}
<!-- 
[The site is under development. Please email [***yuchen.lin@usc.edu***] if you have any questions.](){: .btn .btn-red .fs-4 target="_blank"} -->

