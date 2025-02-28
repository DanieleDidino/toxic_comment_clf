# Train models for toxic language classification 

The goal is to build a small model that does a good job at classifing the toxic language.
It should be able to run on small devices.

This is a work in progress repo!!

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


## Model

Since the train set is about 100K samples and the text is relatively short, a GRU is probably better thatn a Tranformer?


## Streamlit

Community cloud resources and limits (as of February 2024) are approximately as follows:

- CPU: 0.078 cores minimum, 2 cores maximum
- Memory: 690MB minimum, 2.7GBs maximum
- Storage: No minimum, 50GB maximum

