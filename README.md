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

**CNN+GRU** model:

++++++++++++++++++++++++++++++++++++++++++++++++++ TO EDIT (START) ++++++++++++++++++++++++++++++++++++++++++++++++++

- Feature Extraction: CNN acts as a feature extractor, detecting local n-gram patterns before passing them to GRU. This helps GRU focus on longer-term dependencies rather than processing raw word embeddings.
- Efficiency: CNN reduces the sequence length (via pooling), making the GRU process fewer time steps. This speeds up training while retaining important contextual information.
- Empirical Results: Many studies show that CNN+RNN architectures (CNN first, then GRU) outperform RNN-CNN models in text classification.

The GRU+CNN architecture represents a hybrid approach that aims to leverage the strengths of both GRUs and CNNs

GRU-CNN Architecture:
- CNN layer captures the local features and patterns within the text (i.e., n-grams).
- GRU layer is effective at capturing sequential dependencies and long-range contextual information.
- CNN+GRU allows the model to learn both local and global features, improving accuracy.

Main Layers:
- Word embedding: represent the text.
- CNN layer: extract local features.
- GRU layer: processes the sequence and capture contextual information.
- Fully connected: classification layer (6 classes).

###################
Brief description

1. CNN Output Processing:
   - CNN layers extract local features from the word embeddings.
   - The outputs of the convolutional layers are then max-pooled, which essentially selects the most important features from each filter.
   - These max-pooled outputs from the different CNN filters (with varying kernel sizes).

2. Combination with GRU:
   - The output from the CNN layer is sent into the GRU.
   - This allows the GRU to process the local features that the CNN has extracted, and to capture the sequential dependencies between those local features.
   - The output of the GRU is then fed into the final fully connected layer. Therefore the information gained from the CNN, is then passed into the GRU, and then the final fully connected layer.

Sum up:
- The CNN acts as a feature extractor, identifying important local patterns.
- The GRU then takes those extracted local patterns and processes them sequentially, to understand the context of the text.

++++++++++++++++++++++++++++++++++++++++++++++++++ TO EDIT (END) ++++++++++++++++++++++++++++++++++++++++++++++++++

## Streamlit

Community cloud resources and limits (as of February 2024) are approximately as follows:

- CPU: 0.078 cores minimum, 2 cores maximum
- Memory: 690MB minimum, 2.7GBs maximum
- Storage: No minimum, 50GB maximum

