# VALUATION OF AN INTEREST RATE SWAP (IRS)
# Manuel de la Llave Roselló
# Máster Banca y finanzas cuantitativas
# manudela@ucm.es
# 2021

import pandas as pd
from datetime import datetime, date
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scripts.irs_valuation import InterestRateSwap as IRS

############################################
# EXERCISE 1: INTEREST RATE SWAP VALUATION #
############################################
# Importing data
fixed = pd.read_excel(r"C:\Users\manu_\Documents\Clase\Máster\2º año\Seminario finanzas\Práctica\Práctica 1\data"
                       r"\fixed.xlsx", names=['initial_date', 'final_date'])
fixed = fixed.dropna()
float = pd.read_excel(r"C:\Users\manu_\Documents\Clase\Máster\2º año\Seminario finanzas\Práctica\Práctica 1\data"
                       r"\float.xlsx", names=['initial_date', 'final_date'])
float = float.dropna()
fixings = pd.read_excel(r"C:\Users\manu_\Documents\Clase\Máster\2º año\Seminario finanzas\Práctica\Práctica 1\data"
                       r"\fixings.xlsx", names=['date', 'libor3m'])
fixings = fixings.dropna()
fixings.libor3m = fixings.libor3m / 100  # Percentage
libor3m = pd.read_excel(r"C:\Users\manu_\Documents\Clase\Máster\2º año\Seminario finanzas\Práctica\Práctica 1\data"
                       r"\libor3m.xlsx", names=['tenor', 'date', 'yearfrac', 'discount_factor', 'zero_coupon'])
libor3m = libor3m.dropna()
ois = pd.read_excel(r"C:\Users\manu_\Documents\Clase\Máster\2º año\Seminario finanzas\Práctica\Práctica 1\data"
                       r"\ois.xlsx", names=['tenor', 'date', 'yearfrac', 'discount_factor', 'zero_coupon'])
ois = ois.dropna()

# Rename data

fixed_initial_date = fixed.initial_date
fixed_final_date = fixed.final_date
risk_free_discount_factor = ois.discount_factor
risk_free_date = ois.date
float_initial_date = float.initial_date
float_final_date = float.final_date
risky_discount_factor = libor3m.discount_factor
libor3m_date = libor3m.date
fixings_rates = fixings.libor3m
fixings_date = fixings.date

# Contract parameters
notional = 20e6
fixed_coupon = 0.012533
convention = 360 # Act 360
valuation_date = date(2018, 1, 2)
valuation_date = pd.to_datetime(valuation_date)
maturity_date = date(2042, 6, 23)
maturity_date = pd.to_datetime(maturity_date)

# Valuation
swap = IRS(notional, fixed_coupon, valuation_date, maturity_date, convention)
npv = swap.npv(fixed_initial_date, fixed_final_date, risk_free_discount_factor, risk_free_date,
               float_initial_date, float_final_date, risky_discount_factor, libor3m_date,
               fixings_rates, fixings_date)
print('NPV IRS:', npv)

######################################
# EXERCISE 2: OBTAIN A FORWARD CURVE #
######################################

# Importing data
libor6m = pd.read_excel(r"C:\Users\manu_\Documents\Clase\Máster\2º año\Seminario finanzas\Práctica\Práctica 1\data"
                       r"\libor6m.xlsx", names=['tenor', 'date', 'yearfrac', 'discount_factor', 'zero_coupon'])
libor6m = libor6m.dropna()
libor6m_discount_factor = libor6m.discount_factor
libor6m_date = libor6m.date

days = pd.date_range(start="2018-07-04", end="2024-01-04")  # Granularity (daily)

# Lineal interpolation
df_interp_lineal = np.interp(days, libor6m_date, libor6m_discount_factor)
niter = len(df_interp_lineal) - 1
forward_linear = np.zeros(niter)
for i in range(0, niter):
    forward_linear[i] = ((df_interp_lineal[i] / df_interp_lineal[i + 1]) - 1) / (1 / convention)

# Cubic interpolation
# We need transform the dates because interp1d doesn't accept dates
days2 = [np.datetime64(d).astype(datetime) for d in days.values]
libor6m_date2 = [np.datetime64(d).astype(datetime) for d in libor6m_date.values]

cubic_interp = interp1d(libor6m_date2, libor6m_discount_factor, kind='cubic')
df_interp_cubic = cubic_interp(days2)
niter = len(df_interp_cubic) - 1
forward_cubic = np.zeros(niter)
for i in range(0, niter):
    forward_cubic[i] = ((df_interp_cubic[i] / df_interp_cubic[i + 1]) - 1) / (1 / convention)

# Plotting
plot1 = plt.figure(1)
plt.plot(days, df_interp_lineal, '-')
plt.plot(libor6m_date[0:13], libor6m_discount_factor[0:13], 'o')
plt.ylabel('Discount factor LIBOR 6M')
plt.title('Linear interpolation')
plt.legend(['Interpolation', 'Real Value'], loc='best')

plot2 = plt.figure(2)
plt.plot(days[:-1], forward_linear)
plt.ylabel('Daily forward')
plt.title('Linear interpolation')

plot3 = plt.figure(3)
plt.plot(days, df_interp_cubic, '-')
plt.plot(libor6m_date[0:13], libor6m_discount_factor[0:13], 'o')
plt.ylabel('Discount factor LIBOR 6M')
plt.title('Cubic interpolation')
plt.legend(['Interpolation', 'Real Value'], loc='best')

plot4 = plt.figure(4)
plt.plot(days[:-1], forward_cubic)
plt.ylabel('Daily forward')
plt.title('Cubic interpolation')

plot5 = plt.figure(5)
plt.plot(days[:-1], forward_linear, 'o', days[:-1], forward_cubic, 'x')
plt.legend(['Linear Forward', 'Cubic Forward'], loc='best')
