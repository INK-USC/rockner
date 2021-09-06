---
layout: default
title: RockNER
nav_order: 1
description: "RockNER: Evaluating the Robustness of Named Entity Recognition with Natural Adversarial Attacks(ACL21 Findings)"
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


# RockNER: Evaluating the Robustness of Named Entity Recognition with Natural Adversarial Attacks
{: .no_toc style="font-weight: 600; font-family: 'Agency FB Bold', arial;"}
✍️  [***Bill Yuchen Lin***](https://yuchenlin.xyz/){: target="blank"}, [***Wenyang Gao***](){: target="blank"}, [***Jun Yan***](https://junyann.github.io/), [***Ryan Moreno***](https://ryan-moreno.github.io/){: target="blank"}, [***Xiang Ren***](http://www-bcf.usc.edu/~xiangren/){: target="blank"} \
🏢  ***in Proceedings of EMNLP 2021 (short)***

--- 
 

## Quick Links
{: .no_toc}
<!-- {: .fs-7 .fw-700 .text-blue-300 } -->
<span class="fs-4">
[Paper](riddlesense_acl21_paper.pdf){: target="_blank" .btn .btn-green .mr-1 .fs-4}
[Video](https://s3.amazonaws.com/pf-upload-01/u-59356/0/2021-06-25/8f53jb9/riddlesense_acl.mp4){: target="_blank" .btn .btn-green .mr-1 .fs-4}
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

 
<!-- ![intro](images/riddle_intro.png){: style="border: 2px solid black"} -->
<!-- ##  --> 
<!-- ***Abstract.***{: .text-red-100}  -->
 
Neural named entity recognition (NER) models have achieved great performance on many conventional benchmarks such as CoNLL2003 and OntoNotes 5.0.
However, it is not clear whether they are still reliable in realistic applications where entities and/or context words can be out of the distribution of the training data.
It is thus important to audit the robustness of NER systems via natural adversarial attacks.
Most existing methods for generating adversarial attacks in  NLP mainly focus on sentence classification and question answering,
while they don't have special designs to reflect the underlying compositions of the NER examples --- i.e., entity structures and their context words. 
In this paper, 
we focus on creating general natural adversarial examples (i.e., real-world entities and human-readable context) for evaluating the robustness of NER models.

To audit the robustness of named entity recognition (NER) models, we propose RockNER, a simple yet effective method to generate natural adversarial examples via entity-/context-level attacks. 
We apply the proposed attacking method on the OntoNotes 5.0 dataset and create a new benchmark named OntoRock to evaluate the robustness of a variety of existing NER models via a systematic evaluation protocol. 
Our experiments and analysis reveal that even the best model has a significant performance drop.
We find that these models tend to memorize entity patterns instead of reasoning from the context.
Apart from that, we also study the effects of a few data augmentation methods to improve the robustness of NER models. 
<!-- \footnote{Our code and data are publicly available at the project website: \url{https://inklab.usc.edu/rockner}.} -->

## Dataset Format (Todo)

Please download our dataset by filling the [***form***](https://forms.gle/iWdsgN44TeoXW19e6){: target="_blank"} here and the link will show up once you read the disclaimer and submit it. There are five files as follows:

1. `rs_train.jsonl` (3,510 lines)
    - The training data of RockNER.
1. `csqa_train.jsonl` (9,741 lines)
    - The training data of CommonsenseQA.
1. `csqa_rs_train.jsonl` (13,251 lines)
    - The training of of CommonsenseQA + RockNER, i.e., the combination of both.
1. `rs_dev.jsonl` (1,021 lines)
    - The development data of RockNER.
1. `rs_test_hidden.jsonl` (1,184 lines)
    - The test data of RockNER, where the truth answers are hidden.

```json
{   # a particular line in our jsonl file
    "id": "c1235zcx90023230",
    "question": {
        "stem": "My life can be measured in hours. I serve by being devoured. Thin, I am quick. Fat, I am slow. Wind is my foe. What am I?",    # The riddle question.
        "choices": [
            {"label": "A", "text": "paper"},
            {"label": "B", "text": "candle"},   # the correct answer
            {"label": "C", "text": "lamp"},
            {"label": "D", "text": "clock"},
            {"label": "E", "text": "worm"}
        ]
    },
    "answerKey": "B"    # this will be "hidden" in the test data.
}
```


## Leaderboard (Todo)

<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Submitter</th>
    <th>Date</th>
    <th>Training Data</th>
    <th>Acc</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="modelname" style="background-color: #f5f6fa">Humans</td>
    <td class="submitter"  style="background-color: #f5f6fa">-</td>
    <td class="date"  style="background-color: #f5f6fa">-</td>
    <td class="traindata"  style="background-color: #f5f6fa">N/A</td>
    <td class="acc"  style="background-color: #f5f6fa">91.33</td>
  </tr>
  <tr>
    <td class="modelname"><a href="https://arxiv.org/abs/2005.00700" target="_blank">UnifiedQA (T5-3B) </a></td>
    <td class="submitter"><a href="http://inklab.usc.edu" target="_blank">USC-INK</a></td>
    <td class="date">5/30/2021</td>
    <td class="traindata">RS+CSQA</td>
    <td class="acc">68.80</td>
  </tr>
  <tr>
    <td class="modelname"><a href="https://arxiv.org/abs/1909.11942" target="_blank">ALBERT-XXL </a></td>
    <td class="submitter"><a href="http://inklab.usc.edu" target="_blank">USC-INK</a></td>
    <td class="date">5/30/2021</td>
    <td class="traindata">RS+CSQA</td>
    <td class="acc">67.30</td>
  </tr>
  <tr>
    <td class="modelname"><a href="https://arxiv.org/abs/2005.00646" target="_blank">MHGRN (AB-XXL) </a></td>
    <td class="submitter"><a href="http://inklab.usc.edu" target="_blank">USC-INK</a></td>
    <td class="date">5/30/2021</td>
    <td class="traindata">RS+CSQA</td>
    <td class="acc">66.81</td>
  </tr>
  <tr>
    <td class="modelname"><a href="https://arxiv.org/abs/2005.00646" target="_blank">MHGRN (RoBERTa-Large) </a></td>
    <td class="submitter"><a href="http://inklab.usc.edu" target="_blank">USC-INK</a></td>
    <td class="date">5/30/2021</td>
    <td class="traindata">RS+CSQA</td>
    <td class="acc">63.73</td>
  </tr>
  <tr>
    <td class="modelname"><a href="https://arxiv.org/abs/1907.11692" target="_blank">RoBERTa-Large </a></td>
    <td class="submitter"><a href="http://inklab.usc.edu" target="_blank">USC-INK</a></td>
    <td class="date">5/30/2021</td>
    <td class="traindata">RS+CSQA</td>
    <td class="acc">59.82</td>
  </tr>
  <tr>
    <td class="modelname"><a href="https://arxiv.org/abs/1909.02151" target="_blank">KagNet (RoBERTa-Large)</a></td>
    <td class="submitter"><a href="http://inklab.usc.edu" target="_blank">USC-INK</a></td>
    <td class="date">5/30/2021</td>
    <td class="traindata">RS+CSQA</td>
    <td class="acc">59.72</td>
  </tr>
  <tr>
    <td class="modelname"><a href="https://arxiv.org/abs/2005.00700" target="_blank">UnifiedQA (T5-Large)</a></td>
    <td class="submitter"><a href="http://inklab.usc.edu" target="_blank">USC-INK</a></td>
    <td class="date">5/30/2021</td>
    <td class="traindata">RS+CSQA</td>
    <td class="acc">56.57</td>
  </tr>
  <tr>
    <td class="modelname"><a href="https://arxiv.org/abs/1810.04805" target="_blank">BERT-Large</a></td>
    <td class="submitter"><a href="http://inklab.usc.edu" target="_blank">USC-INK</a></td>
    <td class="date">5/30/2021</td>
    <td class="traindata">RS+CSQA</td>
    <td class="acc">54.91</td>
  </tr>
  <tr>
    <td class="modelname"><a href="https://arxiv.org/abs/1810.04805" target="_blank">BERT-Base</a></td>
    <td class="submitter"><a href="http://inklab.usc.edu" target="_blank">USC-INK</a></td>
    <td class="date">5/30/2021</td>
    <td class="traindata">RS+CSQA</td>
    <td class="acc">47.67</td>
  </tr>
  <tr>
    <td class="modelname"  style="background-color: #f5f6fa">Random Guess</td>
    <td class="submitter"  style="background-color: #f5f6fa">-</td>
    <td class="date"  style="background-color: #f5f6fa">-</td>
    <td class="traindata"  style="background-color: #f5f6fa">N/A</td>
    <td class="acc"  style="background-color: #f5f6fa">20.00</td>
  </tr>
</tbody>
</table>

### Submission Guide (Todo)
{: .no_toc}

This is [***an example submission file***](submission_example.jsonl){: target="_blank"}. Please submit your prediction file and information via [***this form***](https://forms.gle/a3yyoxmgj1FoJpMM7){: target="_blank"}.

## Citation

```bibtex
@inproceedings{lin-etal-2021-rockner,
    title = "RockNER: Evaluating the Robustness of Named Entity Recognition with Natural Adversarial Attacks",
    author = "Lin, Bill Yuchen and Gao, Wenyang and Yan, Jun and Moreno, Ryan and Ren, Xiang",
    booktitle = "Proc. of EMNLP (short paper)",
    year = "2021",
    note={to appear}
}
``` 
{: .fs-4}
<!-- 
[The site is under development. Please email [***yuchen.lin@usc.edu***] if you have any questions.](){: .btn .btn-red .fs-4 target="_blank"} -->

