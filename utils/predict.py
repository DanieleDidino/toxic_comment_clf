import numpy as np


def predict(text: str="Text to classify") -> list[float]:
    # ADD HERE MODEL
    probabilities = np.array([0.49, 0.30, 0.50, 0.60, 0.80, 0.99])
    probabilities *= 100 # convert to percentagepass
    return probabilities


def values_to_plot(probabilities: np.array, threshold: float = 0.5) -> np.array:
    """
    If the probability of "toxic" is below the threshold, set the probabilites
    of all the other label to 0
    """
    if probabilities[0] < threshold:
        # If toxic is less than threshold, the other categories are "0"
        new_prob = np.zeros_like(probabilities)
        new_prob[0] = probabilities[0]
    else:
        new_prob = probabilities
    new_prob *= 100 # convert to percentage
    return new_prob
