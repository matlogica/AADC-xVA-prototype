import json
from observable import Observable, Ticker, Observation
from datetime import datetime

class Contract:
    def __add__(self, other):
        if other is None:
            return self
        return Leg(contracts=[self, other])

    def __sub__(self, other):
        if other is None:
            return self
        return Leg(contracts=[self, other.negate()])

class Leg(Contract):
    def __init__(self, contracts) -> None:
        self.contracts = contracts

    def __repr__(self):
        return "\n".join(str(el) for el in self.contracts)

    def negate(self):
        return Leg([cf.negate() for cf in self.contracts])
    
    def to_json(self):
        return [cf.to_json() for cf in self.contracts]

class Cashflow(Contract):
    def __init__(self, observable: Observable, payment_date: datetime, payment_currency: str, notional: float):
        self.observable = observable
        self.payment_date = payment_date
        self.payment_currency = payment_currency
        self.notional = notional

    def negate(self):
        return Cashflow(self.observable, self.payment_date, self.payment_currency, -self.notional)

    def __repr__(self):
        if hasattr(self.observable, "value"):
            return f"{self.observable}\t{self.notional}\t{self.payment_currency}\t{self.payment_date.strftime('%Y-%m-%d')}\t{self.observable.value}"
        return f"{self.observable}\t{self.notional}\t{self.payment_currency}\t{self.payment_date.strftime('%Y-%m-%d')}"
    
    def to_json(self):
        return {
            "observable": str(self.observable),
            "notional": self.notional,
            "payment_currency": self.payment_currency,
            "payment_date": self.payment_date.strftime('%Y-%m-%d')
        }

class Fork(Contract):
    def __init__(self, condition: Observable, contract1: Contract, contract2: Contract) -> None:
        self.condition = condition
        self.contract1 = contract1
        self.contract2 = contract2

    def __repr__(self):
        return f"if {self.condition} is positive then {self.contract1} else\n{self.contract2}"
    
    def to_json(self):
        return {
            "condition": self.condition.to_json(),
            "contract1": self.contract1.to_json(),
            "contract2": self.contract2.to_json()
        }

def callable(decision_obs, cpn_cfs, funding_cfs, cpn_per_call=1, funding_per_call=2, continuation = None):
    if len(decision_obs) == 0:
        return Leg(funding_cfs) - Leg(cpn_cfs) + continuation

    cpn_cfs, last_cpn_cfs = cpn_cfs[:-cpn_per_call], cpn_cfs[-cpn_per_call:]
    funding_cfs, last_funding_cfs = funding_cfs[:-funding_per_call], funding_cfs[-funding_per_call:]
    decision_obs, last_decision_obs = decision_obs[:-1], decision_obs[-1]

    continuation = Fork(last_decision_obs, None, Leg(last_funding_cfs) - Leg(last_cpn_cfs) + continuation)
    return callable(decision_obs, cpn_cfs, funding_cfs, cpn_per_call, funding_per_call, continuation)

if __name__  == "__main__":
    import numpy as np
    import json
    from schedule import schedule
    from dateutil.relativedelta import relativedelta

    start_date = datetime(2024, 7, 10)
    tenor = relativedelta(years=10)

    coupon_leg_schedule = schedule(start_date, tenor, frequency=relativedelta(years=1))
    coupon_fixings = [end - relativedelta(days=2) for _, end in coupon_leg_schedule]
    funding_leg_schedule = schedule(start_date, tenor, frequency=relativedelta(months=6))
    funding_fixings = [start - relativedelta(days=2) for start, _ in funding_leg_schedule]

    call_dates = coupon_fixings[1:]

    CMS20Y = Ticker("EUR.CMS.20Y", "Reuters")
    CMS2Y = Ticker("EUR.CMS.2Y", "Reuters")
    Libor6M = Ticker("EUR.LIBOR.6M", "Reuters")

    funding_cfs = [Cashflow(Observation(Libor6M, fixing) + 0.002, payment_date, "EUR", 1_000_000) for fixing, (_, payment_date) in zip(funding_fixings, funding_leg_schedule)]

    coupons = [np.max(Observation(CMS20Y, fixing) - 2 * Observation(CMS2Y, fixing), 0.0) for fixing in coupon_fixings]
    # tarn = np.cumsum(coupons, 0.0)

    cpn_cfs = [Cashflow(coupon, payment_date, "EUR", 1_000_000) for coupon, (_, payment_date) in zip(coupons, coupon_leg_schedule)]

    call = callable([Observation(Ticker("Call", "Matlogica"), date) for date in call_dates], cpn_cfs, funding_cfs, cpn_per_call=1, funding_per_call=2)

    # coupon_leg = np.sum(cpn_cfs)
    # print(coupon_leg)

    print(call)

#    print(json.dumps(call))

    print("**********")
    print("TARN")

    tarn_obs = np.cumsum(coupons)

    tarn = callable(tarn_obs - 0.1, cpn_cfs, funding_cfs, cpn_per_call=1, funding_per_call=2)
    print(tarn)
