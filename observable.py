import numpy as np
import json
from datetime import datetime

class Ticker:
    def __init__(self, symbol, source):
        self.symbol = symbol        # e.g. "USDJPY" or "Cancel"
        self.source = source        # e.g. "Reuters" or counter-party name for Decision Tickers

    def __repr__(self):
        return self.symbol
    
    def __eq__(self, value: object) -> bool:
        if isinstance(value, Ticker):
            return self.symbol == value.symbol and self.source == value.source
        return False
    
    def __hash__(self):
        return hash((self.symbol, self.source))
    
    def to_json(self):
        return {
            "symbol": self.symbol,
            "source": self.source
        }

class Observable:
    def __add__(self, other):
        if other is None:
            return self
        return ObsOp(dependencies=[self, other], operation='+')

    def __sub__(self, other):
        if other is None:
            return self
        return ObsOp(dependencies=[self, other], operation='-')

    def __mul__(self, other):
        return ObsOp(dependencies=[self, other], operation='*')

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return ObsOp(dependencies=[self, other], operation='/')

    def __array_function__(self, func, types, args, kwargs):
        if func == np.max:
            return ObsOp.max(*args, **kwargs)
        if func == np.min:
            return ObsOp.min(*args, **kwargs)

        return NotImplemented

BINARY_OPERATORS = ['+', '-', '*', '/']


class ObsOp(Observable):
    def __init__(self, dependencies, operation=None, kwargs=None):
        self.dependencies = dependencies
        self.operation = operation
        self.kwargs = kwargs or {}

    def __repr__(self):
        if self.operation in BINARY_OPERATORS:
            return f"({self.dependencies[0]} {self.operation} {self.dependencies[1]})"
        args = ", ".join(map(str, self.dependencies))
        return f"{self.operation}({args}))"
    
    def to_json(self):
        return {
            "operation": self.operation,
            "dependencies": [dep.to_json() for dep in self.dependencies]
        }

    @staticmethod
    def max(*args, **kwargs):
        return ObsOp(dependencies=list(args), operation='max', kwargs=kwargs)
    @staticmethod
    def min(*args, **kwargs):
        return ObsOp(dependencies=list(args), operation='min', kwargs=kwargs)

class Observation(Observable):
    def __init__(self, ticker, fixing_datetime):
        self.ticker = ticker
        self.fixing_datetime = fixing_datetime

    def __repr__(self):
        return f"{self.ticker}({self.fixing_datetime.strftime('%Y-%m-%d')})"
    
    def __eq__(self, value: object) -> bool:
        if isinstance(value, Observation):
            return self.ticker == value.ticker and self.fixing_datetime == value.fixing_datetime
        return False
    
    def __hash__(self):
        return hash((self.ticker, self.fixing_datetime))
    
    def to_json(self):
        return {
            "ticker": self.ticker.to_json(),
            "fixing_datetime": self.fixing_datetime.strftime('%Y-%m-%d')
        }

# Example usage:

if __name__ == "__main__":

    fx = Observation(Ticker("USDJPY", "Reuters"), datetime(2024, 7, 26))
    payoff = np.max(fx - 100, 0)

    print(payoff)
