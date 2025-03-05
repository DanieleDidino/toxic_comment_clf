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

Architecture:
- Embedding Layer: it converts input text into dense vectors of fixed size.
- Convolutional Layer:
  - Conv1D: it applies convolutional filters to the embedded text to capture local patterns (e.g., n-grams).
  - Pooling Layer: Reduces the dimensionality of the convolutional output while retaining important features.
  - (the output from the convolutional layers can be fed into the GRU)
- GRU Layer: it processes the sequential data to capture long-range dependencies and context. 
- Dense Layers: it makes the final classification decision.


Why CNN+GRU:
- CNN: Excels at capturing local patterns and spatial hierarchies in text data. It can identify key phrases or n-grams that are important for classification.
- GRU: Effective at capturing long-range dependencies and sequential information in text, which is crucial for understanding the context and semantics.


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


Hyperparameters:
- Embedding Dimension: size of the embedding vectors (if not pre-trained: typical range 50-300)
- Number of Filters: the number of filters in the convolutional layer (typical range: 64-256)
- Kernel Size: the size of the convolutional kernel. Different kernel sizes capture different n-gram patterns. (typical range: 3-7)
- GRU Hidden Size: the number of features in the hidden state of the GRU. (typical range: 50-200)
- Learning Rate (typical range: 0.0001-0.01)
- Batch Size (typical range: 16-128)
- Dropout Rate (typical range: 0.2-0.5)


++++++++++++++++++++++++++++++++++++++++++++++++++ TO EDIT (END) ++++++++++++++++++++++++++++++++++++++++++++++++++

## Streamlit

``` bach
streamlit run app.py
```

Community cloud resources and limits (as of February 2024) are approximately as follows:

- CPU: 0.078 cores minimum, 2 cores maximum
- Memory: 690MB minimum, 2.7GBs maximum
- Storage: No minimum, 50GB maximum

