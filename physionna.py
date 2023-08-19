import numpy as np
import tensorflow as tf

# Import Sionna
try:
    import sionna
except ImportError as e:
    # Install Sionna if package is not already installed
    import os
    os.system("pip install sionna")
    import sionna
%matplotlib inline
import matplotlib.pyplot as plt


batch_size = 100
num_codewords = 64
info_bit_length = 500

source = sionna.utils.BinarySource()

u = source([batch_size, num_codewords, info_bit_length])
print("Shape of u: ", u.shape)

# initialize an CRC encoder w/ CRC24A
encoder_crc = sionna.fec.crc.CRCEncoder("CRC24A")
decoder_crc = sionna.fec.crc.CRCDecoder(encoder_crc) #link to encoder

c = encoder_crc(u) # returns the list of crc valid
print("Shape of c: ", c.shape)
print("Processed bits: ", np.size(c.numpy()))

u_hat, crc_valid = decoder_crc(c)
print("Shape of u_hat: ", u_hat.shape)
print("Shape of crc_valid: ", crc_valid.shape)

print("Valid CRC check of first codeword: ", crc_valid.numpy()[0,0,0])