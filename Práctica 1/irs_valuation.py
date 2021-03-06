# Manuel de la Llave Roselló
# Máster Banca y finanzas cuantitativas
# manudela@ucm.es
# 2021
import numpy as np
from datetime import date
import dateutil.relativedelta


class InterestRateSwap:
    def __init__(self,
                 input_notional: float,
                 input_fixed_coupon: float,
                 input_valuation_date: date,
                 input_maturity_date: date,
                 input_convention: int):
        """
        We use the contract details as inputs for our IRS.
        Parameters
        ----------
        input_notional: float
            Notional

        input_fixed_coupon: float
            Fixed leg coupon

        input_valuation_date: date
            Present date, the day we want to update IRS' net value

        input_maturity_date: date
            Date when contract ends. Not used here

        input_convention: int
            Either Act 360 or 365
        """
        self.notional = input_notional
        self.coupon = input_fixed_coupon
        self.valuation_date = input_valuation_date
        self.maturity_date = input_maturity_date
        self.act = input_convention

    def npv(self, fixed_initial_date: date,
            fixed_final_date: date,
            risk_free_discount_factor: date,
            tenor_date: date,
            float_initial_date: date,
            float_final_date: date,
            libor_discount_factor: float,
            libor_date: date,
            fixings_rates: float,
            fixings_date: date):
        """
        We first calculate the fixed leg and afterwards the float leg. For doing so, we discount the amount each leg
        pays in every payment date.

        Parameters
        ----------
        fixed_initial_date: date
            Date where the fixed leg starts accrual

        fixed_final_date: date
            Date where the fixed leg is payed

        risk_free_discount_factor: float
            We consider the fixed leg a risk-free counterparty, so we use a risk-free discount factor such as OIS.

        tenor_date: date
            Risk-free discount factor corresponding tenor (1M, 2M, 3M, etc.)

        float_final_date: date
            Date where the float leg starts accrual

        float_final_date: date
            Date where the float leg is payed

        libor_discount_factor: float
            Risky discount factor we use to compute float leg's NPV. In this case, we use LIBOR curves

        fixings_rates: float
            Past LIBOR rates which could be used instead of interpolate all the needed rates

        fixings_date: date
            Date where the fixing rate was known

        Returns
        -------
        Value: float
            IRS' net present value (NPV). Also we show the fixed and float NPV.
        """
        # FIXED LEG
        aux = sum(self.valuation_date >= fixed_final_date)
        delta_fixed = (fixed_final_date[aux:] - fixed_initial_date[aux:]) / self.act
        delta_fixed[aux] = (fixed_final_date[aux] - self.valuation_date) / self.act
        # From time series to float class
        delta_fixed = delta_fixed.dt.seconds + delta_fixed.dt.days * (24 * 60 * 60)
        delta_fixed = delta_fixed / (24 * 60 * 60)
        df_interp = np.interp(fixed_final_date[aux:], tenor_date, risk_free_discount_factor)
        fixed_npv = np.sum(self.coupon * self.notional * delta_fixed * df_interp)
        print('Fixed NPV:', fixed_npv)
        # FLOAT LEG
        aux2 = sum(self.valuation_date >= float_final_date)
        delta_float = (float_final_date[aux2:] - float_initial_date[aux2:]) / self.act
        delta_float[aux2] = (float_final_date[aux2] - self.valuation_date) / self.act
        # From time series to float class
        delta_float = delta_float.dt.seconds + delta_float.dt.days * (24 * 60 * 60)
        delta_float = delta_float / (24 * 60 * 60)
        df_interp_libor = np.interp(float_final_date[aux2 - 1:], libor_date, libor_discount_factor)
        niter = len(df_interp_libor) - 1
        forward = np.zeros(niter)
        for i in range(0, niter):
            forward[i] = ((df_interp_libor[i] / df_interp_libor[i+1]) - 1) / delta_float[aux2 + i]
        index = np.zeros(aux2)
        for j in range(0, aux2):
            index[j] = fixings_date[fixings_date == float_initial_date.values[j]].index[0]
            forward[j] = fixings_rates[index[j]]
        float_discount_factor = np.zeros(niter)
        float_discount_factor[0] = 1 / (1 + delta_float[aux2] * self.coupon)
        my_sum = np.zeros(niter - 1)
        delta_n = [0, delta_float[aux2]]
        for k in range(0, niter - 1):
            my_sum[k] = delta_float[aux2 + k] * float_discount_factor[k]
            float_discount_factor[k + 1] = (1 - self.coupon * sum(my_sum)) / (1 + delta_n[k + 1] * self.coupon)
            delta_n = (float_final_date[aux2:(aux2 + k + 1)] - float_initial_date[aux2]) / self.act
            delta_n = delta_n.dt.seconds + delta_n.dt.days * (24 * 60 * 60)
            delta_n = (delta_n / (24 * 60 * 60))
            # Alternative form with delta_n almost constant
            # float_discount_factor[k + 1] = (1 - self.coupon * sum(my_sum)) / (1 + delta_float[aux2 + k] * self.coupon)
        float_npv = sum(forward * (-self.notional) * delta_float * float_discount_factor)
        print('Float NPV:', float_npv)
        return fixed_npv + float_npv
