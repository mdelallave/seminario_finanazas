# VALUATION OF A CAP
# Manuel de la Llave Roselló
# Máster Banca y finanzas cuantitativas
# manudela@ucm.es
# 2021

import pandas as pd
from datetime import date
from scripts.cap_valuation import cap
import numpy as np

#############################
# EXERCISE 1: CAP VALUATION #
#############################
# Importing data
caplets = pd.read_excel(r"C:\Users\manu_\Documents\Clase\Máster\2º año\Seminario finanzas\Carlos Catalán\Práctica"
                        r"\Práctica 2\data\caplets.xlsx", names=['initial_date', 'final_date'])
caplets = caplets.dropna()
libor3m = pd.read_excel(r"C:\Users\manu_\Documents\Clase\Máster\2º año\Seminario finanzas\Carlos Catalán\Práctica"
                        r"\Práctica 2\data\libor3m.xlsx", names=['tenor', 'date', 'yearfrac', 'discount_factor',
                                                                 'zero_coupon'])
libor3m = libor3m.dropna()
volatilities = pd.read_excel(r"C:\Users\manu_\Documents\Clase\Máster\2º año\Seminario finanzas\Carlos Catalán\Práctica"
                             r"\Práctica 2\data\new_vols.xlsx",
                             names=['tenor', 'date', '-0.50', '-0.25', '-0.13', '0.00',
                                    '0.13', '0.25', '0.50', '1.00', '1.50', '2.00',
                                    '2.50', '3.00', '4.00', '5.00'])
volatilities = volatilities.dropna()
fixings = pd.read_excel(r"C:\Users\manu_\Documents\Clase\Máster\2º año\Seminario finanzas\Carlos Catalán\Práctica"
                        r"\Práctica 2\data\fixings.xlsx", names=['date', 'libor3m'])
fixings = fixings.dropna()
fixings.libor3m = fixings.libor3m / 100  # Percentage

# Rename data
initial_date = caplets.initial_date
final_date = caplets.final_date
discount_factor = libor3m.discount_factor
libor3m_date = libor3m.date
fixings_rates = fixings.libor3m
fixings_date = fixings.date

# Contract parameters
# Cap: USD-3M
notional = 20e6
X = 0.011  # Strike
convention = 360  # Act 360
valuation_date = date(2018, 1, 2)
valuation_date = pd.to_datetime(valuation_date)
maturity_date = date(2042, 6, 23)
maturity_date = pd.to_datetime(maturity_date)
shift = 0.03  # Lognormal model shift

# Implied volatility interpolation
vol_interp_1 = np.interp(final_date, volatilities.date, volatilities.loc[:, "1.00"] / 100)
vol_interp_2 = np.interp(final_date, volatilities.date, volatilities.loc[:, "1.50"] / 100)
# vol_interp_3 =

volatility_strike = vol_interp_1 * 1.1

# Valuation
cap = cap(notional, X, valuation_date, maturity_date, convention)
cap_valuation, caplets = cap.value(initial_date, final_date, volatility_strike, libor3m_date, discount_factor, shift,
                                   fixings_rates, fixings_date)
print('Cap Value:', cap_valuation)

##################################
# EXERCISE 2: IMPLIED VOLATILITY #
##################################

implied_volatility = cap.implied_volatility(initial_date, final_date, volatility_strike, libor3m_date,
                                            discount_factor, shift, fixings_rates, fixings_date)
print('Implied volatility:', implied_volatility)
