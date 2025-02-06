import numpy as np
from observable import Observable, ObsOp, Ticker
from dateutil.relativedelta import relativedelta

######
# Step 4. Define simple t0 market and simulated market. Browninan motion is used for simplicity, but can be easily replaced with more complex models
######

# Market is a simple collection of current and historical index values
class Market:
    def __init__(self, indexes, current_t):
        self.indexes = indexes
        self.t = current_t
    
    def getObservation(self, ticker, obs_datetime):
        if obs_datetime < self.t:
            return self.indexes[ticker.symbol] # historical data
        return self.indexes[ticker.symbol]
    
# Simulated market is a simple extension of Market that allows to simulate index values with shocks
# We also provide definitions for BASIS and CHEBYSHEV polynomials of simulated indexes

class SimulatedMarket:
    def __init__(self, prev_market, shocks, next_t, vols):
        self.prev_market = prev_market
        self.shocks = shocks
        self.t = next_t
        self.indexes = {}
        self.vols = vols

    def getObservation(self, ticker, obs_datetime):

        if self.indexes.get(ticker.symbol) is not None:
            return self.indexes[ticker.symbol]

        if ticker.symbol.startswith("BASIS2_"):
            und = self.getObservation(Ticker(ticker.symbol[7:], ticker.source), obs_datetime)
            return und * und
        if ticker.symbol.startswith("BASIS3_"):
            und = self.getObservation(Ticker(ticker.symbol[7:], ticker.source), obs_datetime)
            return und * und * und
        if ticker.symbol.startswith("BASIS4_"):
            und = self.getObservation(Ticker(ticker.symbol[7:], ticker.source), obs_datetime)
            return und * und * und * und
        
        if ticker.symbol.startswith("CHEB"):
            und = self.getObservation(Ticker(ticker.symbol[6:], ticker.source), obs_datetime)
            normalized = 2 * (und - np.min(und)) / (np.max(und) - np.min(und)) - 1
            if ticker.symbol.startswith("CHEB2"):
                return 2 * normalized * normalized - 1
            if ticker.symbol.startswith("CHEB3"):
                return 4 * normalized * normalized * normalized - 3 * normalized
            if ticker.symbol.startswith("CHEB4"):
                return 8 * normalized * normalized * normalized * normalized - 8 * normalized * normalized + 1
            if ticker.symbol.startswith("CHEB5"):
                return 16 * normalized * normalized * normalized * normalized * normalized - 20 * normalized * normalized * normalized + 5 * normalized
            if ticker.symbol.startswith("CHEB6"):
                return 32 * normalized * normalized * normalized * normalized * normalized * normalized - 48 * normalized * normalized * normalized * normalized + 18 * normalized
            if ticker.symbol.startswith("CHEB7"):
                return 64 * normalized * normalized * normalized * normalized * normalized * normalized * normalized - 112 * normalized * normalized * normalized * normalized * normalized + 56 * normalized * normalized * normalized

        if obs_datetime <= self.prev_market.t:
            return self.prev_market.getObservation(ticker, obs_datetime)
        if obs_datetime > self.t:
            raise ValueError("Observation time is after current time for this Simulated Market")
        if obs_datetime == self.t:
            index_t = self.prev_market.getObservation(ticker, self.prev_market.t)
            # return index_t + self.shocks[ticker.symbol] * np.sqrt((obs_datetime - self.prev_market.t).days / 365)
            # GBM
            days = (obs_datetime - self.prev_market.t).days
            dt = days / 365
            vol = self.vols[ticker.symbol]
            log_return = - 0.5 * vol * vol * dt + vol * np.sqrt(dt) * self.shocks[ticker.symbol]
            val =  index_t * np.exp(log_return)

            self.indexes[ticker.symbol] = val
            return val

        raise NotImplementedError("Interpolation not implemented for SimulatedMarket. Requested observation time is {} ticker is {}. Market time is {} prev market time is {}".format(obs_datetime, ticker, self.t, self.prev_market.t))


def pricing_time_steps(market0):
    today = market0.t
    pricing_time_steps = [today + relativedelta(days=7 * i) for i in range(52 * 5)]
    last_week = pricing_time_steps[-1]
    pricing_time_steps += [last_week + relativedelta(days=30 * i) for i in range(12 * 50)]

    return pricing_time_steps


class Model:
    def __init__(self, vols, pricing_time_steps):
        self.vols = vols
        self.seed = 17
        self.num_paths = 100000
        self.seed_ls = 3
        self.num_paths_ls = 100000
        self.pricing_time_steps = pricing_time_steps
