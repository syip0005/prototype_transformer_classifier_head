# Transformer Classifier Head for NLP

## Description

The aim of this project is to implement a new transformer classifier head for use in 
natural language processing classification problems, and to do so with state of the art
results.

The architecture is similar to that in [[1]](#1) and [[2]](#2) but applied to natural language processing problems
instead of vision problems.

## Training Code

```
python main_train.py --entity=MYNAME --dataset=AG_NEWS --model=bert_classifier \ 
--batch=16 --epochs=15 --lr=1e-6 --weight_decay=0 --bert_type=bert-base-uncased \
--loss=CrossEntropyLoss --learning_rate_style=constant --optimiser=Adam
```

## References

<a id="1">[1]</a> 
Liu et al. (2021). 
Query2label: A simple transformer way to multi-label classification.
arXiv preprint arXiv:2107.10834

<a id="1">[2]</a> 
Radnik et al. (2021). 
ML-Decoder: Scalable and Versatile Classification Head.
arXiv preprint arXiv:2111.12933