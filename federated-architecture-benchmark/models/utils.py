
import torch

def get_model(model_type, **kwargs):
    if model_type == "cnn":
        from models.cnn import CNN
        return CNN(**kwargs)
    elif model_type == "rnn":
        from models.rnn import RNN
        return RNN(**kwargs)
    elif model_type == "lstm":
        from models.rnn import LSTMClassifier
        return LSTMClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
