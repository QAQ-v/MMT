# MMT
An implementation for paper "Multimodal Transformer for Multimodal Machine Translation" (accepted at ACL20). 

# Train and Evaluate
```bash run.sh```

# Pretrained Model and Outputs
The pre-trained models are released [here](https://drive.google.com/file/d/1eMm3hfBqnbUpPXdSV1MWJWUNy3uCmLIb/view?usp=sharing, https://drive.google.com/file/d/1p-8pNuA7DqCuKVD7Vqu8vX36gnvZFWww/view?usp=sharing). The ADAM_acc_99.72_ppl_1.14_e153.pt.translations-test2016 (BLEU 38.7) and ADAM_acc_87.16_ppl_1.74_e22.pt.translations-test2016 (BLEU 39.5) are the corresponding outputs. 

# Citation
```
@inproceedings{yao-wan-2020-multimodal,
    title = "Multimodal Transformer for Multimodal Machine Translation",
    author = "Yao, Shaowei  and
      Wan, Xiaojun",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.400",
    doi = "10.18653/v1/2020.acl-main.400",
    pages = "4346--4350",
    abstract = "Multimodal Machine Translation (MMT) aims to introduce information from other modality, generally static images, to improve the translation quality. Previous works propose various incorporation methods, but most of them do not consider the relative importance of multiple modalities. Equally treating all modalities may encode too much useless information from less important modalities. In this paper, we introduce the multimodal self-attention in Transformer to solve the issues above in MMT. The proposed method learns the representation of images based on the text, which avoids encoding irrelevant information in images. Experiments and visualization analysis demonstrate that our model benefits from visual information and substantially outperforms previous works and competitive baselines in terms of various metrics.",
}
```
