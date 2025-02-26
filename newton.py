import pandas as pd
import numpy as np
import math
from scipy.stats import norm

# Black-Scholes Call Function
def BScall(S, sigma, K, T, r):
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    except Exception as e:
        print(f"Error in BScall: {e}")
        return np.nan

# Vega Function
def vega(S, sigma, K, T, r):
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * norm.pdf(d1) * np.sqrt(T)
    except Exception as e:
        print(f"Error in vega: {e}")
        return 0.0

# Initial Guess Function
def punt_inflexio(S, K, T, r):
    try:
        m = S / (K * math.exp(-r * T))
        return max(0.01, math.sqrt(2 * np.abs(math.log(m)) / T))
    except Exception as e:
        print(f"Error in punt_inflexio: {e}")
        return 0.1

# Newton-Raphson Method for Implied Volatility
def volimplicita(O, S, K, T, r, tol=1e-10, max_iter=700):
    x0 = punt_inflexio(S, K, T, r)
    for _ in range(max_iter):
        try:
            p = BScall(S, x0, K, T, r)
            v = vega(S, x0, K, T, r)

            if v == 0 or np.isnan(v) or v < 1e-06:
                return np.nan

            correction = (p - O) / v
            x0 -= correction

            if x0 < 0.01 or x0 > 5:
                return np.nan

            if abs(p - O) < tol:
                return x0
        except Exception as e:
            print(f"Error in volimplicita iteration: {e}")
            return np.nan
    return np.nan

# Load Data File
dir_arxiu = '/home/luis/Escritorio/material tfg/datos spx/datos_cboe_with_heston.csv'
try:
    dades_opcions = pd.read_csv(dir_arxiu)
except Exception as e:
    print(f"Error loading file: {e}")
    raise

# Constant Interest Rate
r = 0.03

# Calculate Implied Volatility for Calls
vol_imp_calls = []
for _, row in dades_opcions.iterrows():
    try:
        S = row['Preu Stock']
        K = row['Strike']
        T = row['Temps al venciment (anys)']
        C = row['Heston Call Price']
        
        # Ensure all necessary inputs are valid
        if S > 0 and K > 0 and T > 0 and C > 0:
            call_iv = volimplicita(C, S, K, T, r)
        else:
            call_iv = np.nan
        
        vol_imp_calls.append(call_iv)
    except Exception as e:
        print(f"Error processing row: {e}")
        vol_imp_calls.append(np.nan)

# Add Results to DataFrame
dades_opcions['Volatilitat implicita'] = vol_imp_calls

# Save New File
dir_arxiu_sortida = '/home/luis/Escritorio/material tfg/datos spx/volheston.csv'
try:
    dades_opcions.to_csv(dir_arxiu_sortida, index=False)
    print(f"Archivo procesado y guardado en: {dir_arxiu_sortida}")
except Exception as e:
    print(f"Error saving file: {e}")
