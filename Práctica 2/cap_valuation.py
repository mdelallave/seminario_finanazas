# Manuel de la Llave Roselló
# Máster Banca y finanzas cuantitativas
# manudela@ucm.es
# 2021
import numpy as np
from datetime import date
from scipy.stats import norm


class cap:
    def __init__(self,
                 input_notional: float,
                 input_strike: float,
                 input_valuation_date: date,
                 input_maturity_date: date,
                 input_convention: int):
        """
        We use the contract details as inputs for our Cap.
        Parameters
        ----------
        input_notional: float
            Notional

        input_strike: float
            Strike

        input_valuation_date: date
            Present date, the day we want to update Cap's net value

        input_maturity_date: date
            Date when contract ends. Not used here

        input_convention: int
            Either Act 360 or 365
        """
        self.notional = input_notional
        self.strike = input_strike
        self.valuation_date = input_valuation_date
        self.maturity_date = input_maturity_date
        self.act = input_convention

    def value(self, initial_date: date,
              final_date: date,
              volatility_strike: float,
              libor_date: date,
              libor_discount_factor: float,
              shift: float,
              fixings_rates: float,
              fixings_date: date):
        """
        We discount the amount the Cap pays in every payment date if the interest rate is higher than the strike

        Parameters
        ----------
        initial_date: date
            Date where the cap starts accrual

        final_date: date
            Date where the cap is payed

        volatility_strike: float
            Implied volatility as strike value (Interest Rate)

        libor_date: date
            Discount factor corresponding tenor (1M, 2M, 3M, etc.)

        libor_discount_factor: float
            Risky discount factor we use to compute Cap's NPV. In this case, we use LIBOR curves

        shift: float
            Lognormal model shift size

        fixings_rates: float
            Past LIBOR rates which could be used instead of interpolate all the needed rates

        fixings_date: date
            Date where the fixing rate was known

        Returns
        -------
        Value: float
            Cap's net present value (NPV).
        """
        aux2 = sum(self.valuation_date >= final_date)
        delta = (final_date[aux2:] - initial_date[aux2:]) / self.act
        delta[aux2] = (final_date[aux2] - self.valuation_date) / self.act
        # From time series to float class
        delta = delta.dt.seconds + delta.dt.days * (24 * 60 * 60)
        delta = delta / (24 * 60 * 60)
        df_interp_libor = np.interp(final_date[aux2 - 1:], libor_date, libor_discount_factor)
        niter = len(volatility_strike) - aux2
        forward = np.zeros(niter)
        index = np.zeros(aux2)
        for j in range(0, aux2):
            index[j] = fixings_date[fixings_date == initial_date.values[j]].index[0]
            forward[j] = fixings_rates[index[j]]
        nsim = 1000000  # 1e6
        for i in range(aux2 + 1, niter):
            forward[i] = np.mean(shift + (forward[aux2] - shift) * np.exp(volatility_strike[i] *
                                                                          np.random.normal(0, delta[i], nsim) - 0.5
                                                                          * volatility_strike[i] ** 2 * delta[i]))
        d1 = (np.log((forward - shift) / (self.strike - shift)) + 0.5 * volatility_strike[2:] ** 2 * (delta)) / \
             (volatility_strike[2:] * np.sqrt(delta))
        d2 = (np.log((forward - shift) / (self.strike - shift)) - 0.5 * volatility_strike[2:] ** 2 * (delta)) / \
             (volatility_strike[2:] * np.sqrt(delta))
        N1_cdf = norm.cdf(d1, 0, 1)
        N2_cdf = norm.cdf(d2, 0, 1)
        caplets = ((forward - shift) * N1_cdf - (self.strike - shift) * N2_cdf) * self.notional * \
                  df_interp_libor[1:]
        return sum(caplets), caplets

    def implied_volatility(self, initial_date: date,
                           final_date: date,
                           volatility_strike: float,
                           libor_date: date,
                           libor_discount_factor: float,
                           shift: float,
                           fixings_rates: float,
                           fixings_date: date):
        """
        Implied volatility of a Cap following a normal model

        Parameters
              ----------
              initial_date: date
                  Date where the cap starts accrual

              final_date: date
                  Date where the cap is payed

              volatility_strike: float
                  Implied volatility as strike value (Interest Rate)

              libor_date: date
                  Discount factor corresponding tenor (1M, 2M, 3M, etc.)

              libor_discount_factor: float
                  Risky discount factor we use to compute Cap's NPV. In this case, we use LIBOR curves

              shift: float
                  Lognormal model shift size

              fixings_rates: float
                  Past LIBOR rates which could be used instead of interpolate all the needed rates

              fixings_date: date
                  Date where the fixing rate was known

              Returns
              -------
              Value: float
                  Implied volatility.
              """
        aux2 = sum(self.valuation_date >= final_date)
        delta = (final_date[aux2:] - initial_date[aux2:]) / self.act
        delta[aux2] = (final_date[aux2] - self.valuation_date) / self.act
        # From time series to float class
        delta = delta.dt.seconds + delta.dt.days * (24 * 60 * 60)
        delta = delta / (24 * 60 * 60)
        df_interp_libor = np.interp(final_date[aux2 - 1:], libor_date, libor_discount_factor)
        niter = len(delta)
        forward = np.zeros(niter)
        index = np.zeros(aux2)
        for j in range(0, aux2):
            index[j] = fixings_date[fixings_date == initial_date.values[j]].index[0]
            forward[j] = fixings_rates[index[j]]
        nsim = 1000000
        for i in range(aux2, niter):
            forward[i] = np.mean(shift + (forward[aux2 - 1] - shift) * np.exp(volatility_strike[i] *
                                                                              np.random.normal(0, delta[i], nsim) - 0.5
                                                                              * volatility_strike[i] ** 2 * delta[i]))
        d1 = (np.log((forward - shift) / (self.strike - shift)) + 0.5 * volatility_strike[2:] ** 2 * (delta)) / \
             (volatility_strike[2:] * np.sqrt(delta))
        d2 = (np.log((forward - shift) / (self.strike - shift)) - 0.5 * volatility_strike[2:] ** 2 * (delta)) / \
             (volatility_strike[2:] * np.sqrt(delta))
        N1_cdf = norm.cdf(d1, 0, 1)
        N2_cdf = norm.cdf(d2, 0, 1)
        caplets = np.zeros(niter)
        imp_vol = np.zeros(niter)
        TOLABS = 1e-6
        MAXITER = 100
        h = 1e-6
        for f in range(0, niter):
            caplets[f] = ((forward[f] - shift) * N1_cdf[f] - (self.strike - shift) * N2_cdf[f]) * self.notional * \
                         df_interp_libor[f]
            # Newton-Raphson Method
            imp_vol[f] = 0.005  # Initial guess
            dSigma = 10 * TOLABS  # Enter loop for the first time
            nIter = 0
            W = np.mean(np.random.normal(0, delta[aux2 + f], nsim))
            while nIter < MAXITER and np.abs(dSigma) > TOLABS:
                nIter = nIter + 1
                d = (forward[f] - self.strike) / (imp_vol[f] * np.sqrt(delta[aux2 + f]))
                N_cdf = norm.cdf(d, 0, 1)
                N_pdf = norm.pdf(d, 0, 1)
                normal_caplet = ((forward[f] - self.strike) * N_cdf + imp_vol[f] * np.sqrt(delta[aux2 + f]) * N_pdf) \
                                * df_interp_libor[f] * self.notional
                # Numerical derivative (vega)
                forward2 = shift + (forward[aux2 - 1] - shift) * np.exp((imp_vol[f] + h) * W - 0.5
                                                                        * (imp_vol[f] + h) ** 2 * delta[
                                                                            aux2 + f])
                d2 = (forward2 - self.strike) / ((imp_vol[f] + h) * np.sqrt(delta[aux2 + f]))
                N_cdf2 = norm.cdf(d2, 0, 1)
                N_pdf2 = norm.pdf(d2, 0, 1)
                vega_caplet = (((forward2 - self.strike) * N_cdf2 + (imp_vol[f] + h) * np.sqrt(
                    delta[aux2 + f]) * N_pdf2)
                               * df_interp_libor[f] * self.notional - normal_caplet) / h
                dSigma = (normal_caplet - caplets[f]) / vega_caplet
                imp_vol[f] = imp_vol[f] - dSigma
        return imp_vol
