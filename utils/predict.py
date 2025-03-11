import numpy as np
from utils.tokenizer import tokenizer


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict_onnx(text, vocab, session):
    MAX_LEN = 150 # expected input length
    input = tokenizer(text, vocab, max_len=MAX_LEN)
    # Get ONNX model predictions
    output = session.run(None, {"input": input})[0]
    output = sigmoid(output)
    return output[0]


def prediction_toxic(probability_toxic, threshold):
    threshold *= 100  # convert to percentage
    if probability_toxic > threshold:
        pred = "TOXIC"
    else:
        pred = "NOT TOXIC"
    return pred
