from datetime import datetime
from observable import Ticker, Observation
from contract import Cashflow, callable
import numpy as np
from schedule import schedule
from dateutil.relativedelta import relativedelta

if __name__  == "__main__":
    today = datetime(2024, 9, 25)
    start_date = today + relativedelta(days=7)
    tenor = relativedelta(years=10)

    # Funding leg
    funding_leg_schedule = schedule(start_date, tenor, frequency=relativedelta(months=6))
    funding_fixings = [start - relativedelta(days=2) for start, _ in funding_leg_schedule]

    funding_index = Ticker("JPY.Libor.6m", "Reuters")
    funding_cfs = [Cashflow(Observation(funding_index, fixing) + 0.0015, payment_date, "JPY", 100_000_000) for fixing, (_, payment_date) in zip(funding_fixings, funding_leg_schedule)]

    # Structured leg
    coupon_leg_schedule = schedule(start_date, tenor, frequency=relativedelta(years=1))
    coupon_fixings = [end - relativedelta(days=2) for _, end in coupon_leg_schedule]

    fx = Ticker("USDJPY", "Reuters")
    coupons = [np.max(Observation(fx, fixing) - 135, 0.0) for fixing in coupon_fixings]
    cpn_cfs = [Cashflow(coupon, payment_date, "JPY", 100_000_000) for coupon, (_, payment_date) in zip(coupons, coupon_leg_schedule)]

    # Call
    call_dates = coupon_fixings[1:-1]
    call_obs = [Observation(Ticker("Call", "Matlogica"), date) for date in call_dates]

    prdc = callable(call_obs, cpn_cfs, funding_cfs, cpn_per_call=1, funding_per_call=2)
    print(prdc)
