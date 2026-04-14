import numpy as np


def bin2dec_ndc(input):
    temp_piece = 0
    l = len(input)
    for k in range(0, l):
        temp_piece += input[k] << ((l - 1) - k)
    return temp_piece

def dec2bin_ndc(d, size):
    z = format(d, 'b').zfill(size)
    return z
