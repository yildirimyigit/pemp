import torch
import math


def generate_positional_encoding(length=200, d_model=10, frequency_scaler=0.2):  # d_model: dimension of encoding space
    pos = torch.arange(length).unsqueeze(1).float()  # (length, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2,)
    div_term *= frequency_scaler  # Scale the frequency

    pe = torch.zeros(length, d_model)  # (length, d_model)
    pe[:, 0::2] = torch.sin(pos * div_term)
    if d_model % 2 == 1:
        pe[:, 1::2] = torch.cos(pos * div_term[:-1])
    else:
        pe[:, 1::2] = torch.cos(pos * div_term)
    
    return pe


def generate_encoding(length=200, d_model=10):
    enc = torch.zeros(d_model, length)
    max_num = length

    i, j = 0, 1

    for i in range(d_model):
        increment = (i + 1) * 5
        
        while True:
            while enc[i, j-1] <= max_num-increment and j < length:
                enc[i, j] = enc[i, j-1] + increment
                j += 1
            while enc[i, j-1] >= increment and j < length:
                enc[i, j] = enc[i, j-1] - increment
                j += 1

            if j < length:
                continue
            else:
                j = 1
                break
    return enc.T/length


def generate_positional_encoding_for_phase(phase, d_model=10, frequency_scaler=0.2):  # d_model: dimension of encoding space
    length = phase.shape[0]
    # pos = torch.arange(length).unsqueeze(1).float()  # (length, 1)
    pos = phase
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2,)
    div_term *= frequency_scaler  # Scale the frequency

    pe = torch.zeros(length, d_model)  # (length, d_model)
    pe[:, 0::2] = torch.sin(pos * div_term)
    if d_model % 2 == 1:
        pe[:, 1::2] = torch.cos(pos * div_term[:-1])
    else:
        pe[:, 1::2] = torch.cos(pos * div_term)
    
    return pe



pes = [generate_positional_encoding, generate_encoding]