# Train models for toxic language classification 

## Multilabel classification

**Multilabel classification** is a type of classification task where each instance can be assigned multiple labels simultaneously.
Namely, multilabel classification allows for the possibility that an instance can belong to multiple classes at once.

Key aspect:
- **Multiple Labels**: Each instance can have more than one label. For example, a toxic comment might be tagged with both "threat" and "identity_hate".
- **Independent Labels**: The labels are not mutually exclusive, that is an instance can belong to none, one, or several classes.

## Data

The data is from [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).
The dataset includes a large number of Wikipedia comments which have been labelled by human raters for toxic behaviour.

Variables:

- **id**: comment id
- **comment_text**: comment
- **toxic**: binary - true/false
- **severe_toxic**: binary - true/false
- **obscene**: binary - true/false
- **threat**: binary - true/false
- **insult**: binary - true/false
- **identity_hate**: binary - true/false
