# MCC entropy

import pandas as pd
import numpy as np
from compute_ranks_and_copula_dens_multigrid_v2 import compute_ranks_and_copula_dens_multigrid_v2
from compute_copula_entropies import compute_copula_entropies

def MCC_entropy(x, y): 
    # Multiscale Symmetrized Cumulative Copula Entropy

    if isinstance(x, pd.Series):
        x = x.to_numpy()
        
    x = x.astype(float)
        
    if isinstance(y, pd.Series):
        y = y.to_numpy()

    y = y.astype(float)
        
    n = len(x)

    grid_4, grid_8, grid_16, grid_32, grid_64, grid_128, grid_256, grid_512, grid_1024  = compute_ranks_and_copula_dens_multigrid_v2(x, y, n)    
    
    grid_4 += np.finfo(np.float64).tiny 
    grid_8 += np.finfo(np.float64).tiny 
    grid_16 += np.finfo(np.float64).tiny 
    grid_32 += np.finfo(np.float64).tiny 
    grid_64 += np.finfo(np.float64).tiny 
    grid_128 += np.finfo(np.float64).tiny 
    grid_256 += np.finfo(np.float64).tiny 
    grid_512 += np.finfo(np.float64).tiny 
    grid_1024 += np.finfo(np.float64).tiny 

    S_p_p_4, S_m_p_4, S_p_m_4, S_m_m_4 = compute_copula_entropies(grid_4, 4)
    S_p_p_8, S_m_p_8, S_p_m_8, S_m_m_8 = compute_copula_entropies(grid_8, 8)
    S_p_p_16, S_m_p_16, S_p_m_16, S_m_m_16 = compute_copula_entropies(grid_16, 16)
    S_p_p_32, S_m_p_32, S_p_m_32, S_m_m_32 = compute_copula_entropies(grid_32, 32)
    S_p_p_64, S_m_p_64, S_p_m_64, S_m_m_64 = compute_copula_entropies(grid_64, 64)
    S_p_p_128, S_m_p_128, S_p_m_128, S_m_m_128 = compute_copula_entropies(grid_128, 128)
    S_p_p_256, S_m_p_256, S_p_m_256, S_m_m_256 = compute_copula_entropies(grid_256, 256)
    S_p_p_512, S_m_p_512, S_p_m_512, S_m_m_512 = compute_copula_entropies(grid_512, 512)
    S_p_p_1024, S_m_p_1024, S_p_m_1024, S_m_m_1024 = compute_copula_entropies(grid_1024, 1024)
    
    SCCE_4 = S_p_p_4 + S_m_m_4 - S_p_m_4 - S_m_p_4 
    SCCE_8 = S_p_p_8 + S_m_m_8 - S_p_m_8 - S_m_p_8
    SCCE_16 = S_p_p_16 + S_m_m_16 - S_p_m_16 - S_m_p_16
    SCCE_32 = S_p_p_32 + S_m_m_32 - S_p_m_32 - S_m_p_32
    SCCE_64 = S_p_p_64 + S_m_m_64 - S_p_m_64 - S_m_p_64
    SCCE_128 = S_p_p_128 + S_m_m_128 - S_p_m_128 - S_m_p_128
    SCCE_256 = S_p_p_256 + S_m_m_256 - S_p_m_256 - S_m_p_256
    SCCE_512 = S_p_p_512 + S_m_m_512 - S_p_m_512 - S_m_p_512
    SCCE_1024 = S_p_p_1024 + S_m_m_1024 - S_p_m_1024 - S_m_p_1024
    
    SCCE_array = np.array([SCCE_4, SCCE_8, SCCE_16, SCCE_32, SCCE_64, SCCE_128,SCCE_256,SCCE_512,SCCE_1024]) 
    SCCE_abs_array = np.abs(SCCE_array)
    SCCE_argmax = np.argmax(SCCE_abs_array)
    SCCE_max = SCCE_array[SCCE_argmax]

    SCCE_sign = np.sign(SCCE_max)
    SCCE_val = np.abs(SCCE_max)
    
    #a = 2.99319243  # values to fit n = 5000
    #b =  2.70587669
    #c = 0.28656317
    #d = -3.27931520
    a =  2.97579373   # values to fit n = 1000
    b = 2.81571180 
    c = 0.18098956 
    d = -3.15693695 
    
    SCCE_norm = SCCE_sign * (a + b*SCCE_val + c*np.exp(SCCE_val) + d*np.exp(SCCE_val**3))
    if SCCE_norm > 1.0:
        SCCE_norm = 1.0
    if SCCE_norm < - 1.0:
        SCCE_norm = - 1.0        
    
    return (np.round(SCCE_norm, 5).item())

