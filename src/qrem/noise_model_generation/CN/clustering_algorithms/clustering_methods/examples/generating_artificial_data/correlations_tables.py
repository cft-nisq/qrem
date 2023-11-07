import numpy as np
from scipy.stats import norm
import random


""" Generates block correlation table with constant values within each block and constant bias value outside
    all blocks (clusters).
"""
def generate_simple_correlations_table(block_sizes, values, bias):
    total_size = 0
    for size in block_sizes:
        total_size+=size

    correlations_table=np.full((total_size, total_size), bias)

    shift=0
    for block_no in range(len(block_sizes)):
        for local_index_i in range(block_sizes[block_no]):
            for local_index_j in range(block_sizes[block_no]):
                if local_index_i != local_index_j:
                    correlations_table[local_index_i+shift][local_index_j+shift]=values[block_no]
                else:
                    correlations_table[local_index_i+shift][local_index_j+shift]=0
        shift+= block_sizes[block_no]

    return correlations_table

""" Generates block correlation table with off-diagonal values within each block drawn from gaussian distribution centered
 halfway between treshold and 1. Values outside all block are drawn from gaussian distribution centered
 halfway between 0 and treshold. Values outside the range [0,1] are rejected and chosen until they are in [0,1].
Overlap parameter regulates the (common) variance of both gaussians so that the max from probabilities of getting off-block value
 larger than treshold and in-block value below the treshold is = overlap.
"""
def generate_simple_correlations_table_gaussian(block_sizes, treshold, overlap):
    mu_block = 0.5 + treshold/2.0
    mu_rest = treshold/2.0

    sigma=0
    if treshold < 0.5:
        sigma = (treshold-mu_rest)/norm.ppf(1-overlap)
    else:
        sigma = (treshold-mu_block)/norm.ppf(overlap)

    total_size = 0
    for size in block_sizes:
        total_size += size

    correlations_table = np.full((total_size, total_size), 0.0)

    shift = 0
    for block_no in range(len(block_sizes)):
        for local_index_i in range(block_sizes[block_no]):
            for local_index_j in range(block_sizes[block_no]):
                if local_index_i != local_index_j:
                    while True:
                        val=random.gauss(mu_block, sigma)
                        correlations_table[local_index_i + shift][local_index_j + shift] = val
                        if val >= 0.0 and val <=1.0:
                            break
                else:
                    correlations_table[local_index_i + shift][local_index_j + shift] = 0


        for local_index_i in range(block_sizes[block_no]):
            for local_index_j in range(block_sizes[block_no],total_size-shift):
                while True:
                    val = random.gauss(mu_rest, sigma)
                    correlations_table[local_index_i + shift][local_index_j + shift] = val
                    if val >= 0.0 and val <=1.0:
                        break

        for local_index_i in range(block_sizes[block_no],total_size-shift):
            for local_index_j in range(block_sizes[block_no]):
                while True:
                    val = random.gauss(mu_rest, sigma)
                    correlations_table[local_index_i + shift][local_index_j + shift] = val
                    if val >= 0.0 and val <=1.0:
                        break
        shift += block_sizes[block_no]
    return correlations_table