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

print_offset = 0

class Leg(Contract):
    def __init__(self, contracts) -> None:
        self.contracts = contracts

    def __repr__(self):
        result = "\n".join(str(el) for el in self.contracts)
        return result

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
        global print_offset
        offset = " " * print_offset
        return f"{offset}{self.observable}\t{self.notional}\t{self.payment_currency}\t{self.payment_date.strftime('%Y-%m-%d')}"

    def to_json(self):
        return {
            "observable": str(self.observable),
            "notional": self.notional,
            "payment_currency": self.payment_currency,
            "payment_date": self.payment_date.strftime('%Y-%m-%d')
        }

class Option(Contract):
    def __init__(self, condition: Observable, contract1: Contract, contract2: Contract) -> None:
        self.condition = condition
        self.contract1 = contract1
        self.contract2 = contract2

    def __repr__(self):
        global print_offset
        offset = " " * print_offset
        result = f"{offset}if {self.condition} is positive then\n"
        print_offset += 4
        if not isinstance(self.contract1, Contract):
            result += " " * print_offset
        result += f"{self.contract1}\n{offset}else\n"
        if not isinstance(self.contract2, Contract):
            result += " " * print_offset
        result += f"{self.contract2}"
        print_offset -= 4
        return result

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

    continuation = Option(last_decision_obs, None, Leg(last_funding_cfs) - Leg(last_cpn_cfs) + continuation)
    return callable(decision_obs, cpn_cfs, funding_cfs, cpn_per_call, funding_per_call, continuation)

if __name__  == "__main__":
    import numpy as np
    import json
    from schedule import schedule
    from dateutil.relativedelta import relativedelta

    strike = 100.0
    expiry = datetime(2024, 7, 10)
    ccy = "EUR"
    notional = 1_000_000
    ticker = Ticker("AAPL", "Yahoo")
    european_put_option = Cashflow(np.max(Observation(ticker, expiry) - strike, 0), expiry, ccy, notional)

    # european_call_option using Option in "autocallable" style
    call_flag = Observation(ticker, expiry) - strike
    european_call_option = Option(call_flag, Cashflow(Observation(ticker, expiry), expiry, ccy, notional) - Cashflow(strike, expiry, ccy, notional), None)

    # european_call_option using Option with counterparty decision
    # For pricing purposes, the decision can be modeled using Longstaff-Schwartz algorithm
    cpty_decision = Observation(Ticker("TradeID", "CPTY"), expiry)
    european_call_option2 = Option(cpty_decision, Cashflow(Observation(ticker, expiry), expiry, ccy, notional) - Cashflow(strike, expiry, ccy, notional), None)
    print(european_call_option2)

    start_date = datetime(2024, 7, 10)
    tenor = relativedelta(years=10)

    coupon_leg_schedule = schedule(start_date, tenor, frequency=relativedelta(years=1))
    coupon_fixings = [end - relativedelta(days=2) for _, end in coupon_leg_schedule]
    funding_leg_schedule = schedule(start_date, tenor, frequency=relativedelta(months=6))
    funding_fixings = [start - relativedelta(days=2) for start, _ in funding_leg_schedule]

    call_dates = coupon_fixings[1:-1]

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

    tarn_obs = np.cumsum(coupons)[:-1]

    tarn = callable(tarn_obs - 0.1, cpn_cfs, funding_cfs, cpn_per_call=1, funding_per_call=2)
    print(tarn)
