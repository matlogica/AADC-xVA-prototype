from datetime import datetime
from dateutil.relativedelta import relativedelta
from contract import Contract, Cashflow, Leg, Option
from scipy.stats import norm

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

import numpy as np
import copy

from market import Market, SimulatedMarket, Model

# WorkFlow:

# Step 1. Define simple framework to specify trade cashflows, autocall decisions and callable trades
# Step 2. Create a portfolio of 3 trades: 2 European options and 1 callable trade. Callable trade has single call decision and essentially is a European option
# Step 3. Define a function to perform fuzzification on autocall contracts. This function keeps only call decisions
# Step 4. Define simple t0 market and simulated market. Browninan motion is used for simplicity, but can be easily replaced with more complex models
# Step 5. Define a function to update observables in contracts. This function is used to update observables in contracts with new market data (historical or simulated)
# Step 6. Fuzzify autocallable trades
# Step 7. Check if any trades are callable, if yes, run LS regression to estimate basis coefficients and convert to fuzzified contracts
# Step 8. t0 Price portfolio of trades using Monte Carlo simulation. This step can be done path-wise
# Step 9. Run Monte Carlo to collect cashflows from all trades for XVA calculations by regression
# Step 10. Run LS regression to estimate future MtM of trades
#TODO NEXT:
# Step 11. Run Monte Carlo to simulate portfolio MtM and calculate CVA/DVA/FVA without collateral modelling
# Step 12. Run Monte Carlo to simulate portfolio MtM and calculate CVA/DVA/FVA with collateral modelling
# Step 13. Run Monte Carlo to simulate portfolio MtM and calculate CVA/DVA/FVA with advanced collateral modelling

#TODO AADC:
# Add AADC recording to accelerate Step 7 and Step 8
# Add AADC recording to accelerate Step 9 to Step 13
# Demostrate how AADC recording can be used to record Trade cashflows generation. I.e. function form market observables to trade cashflows


# observable defines simple framework to specify trade cashflows
from observable import Observable, Ticker, Observation, ObsOp

#####
# Step 1. Define simple framework to specify trade cashflows, autocall decisions and callable trades
#####
# Main concepts:
# Ticker - represents id of an index on a Market. Holds symbol("USDJPY") and source("Reuters") fields
# Observation - represents an observation of a ticker at a certain datetime. Holds ticker and fixing_datetime fields
# Observation objects can be combined using math operators to create complex expressions. Basic math operators are provided
#Example:
# obs1 = Observation(Ticker("USDJPY", "Reuters"), datetime(2024, 7, 26))
# obs2 = Observation(Ticker("EURUSD", "Reuters"), datetime(2024, 7, 26))
# obs3 = obs1 + obs2
# obs4 = np.max(obs1 , obs2)

# Note that obs3 and obs4 are not evaluated at this point. They are just expressions that can be evaluated later

# Contract represent a trade or a part of a trade. It can be a simple Cashflow, a Leg of Cashflows or a Option of two Contracts
# Cashflow - represents a simple cashflow. Holds observable, payment_date, payment_currency and notional fields
# Leg - represents a list of contracts. Holds contracts field
# Option - represents a decision point in a trade. Holds condition, contract1 and contract2 fields
# Option condition is an Observation that represents a decision point in a trade. If the condition is positive, contract1 is executed, otherwise contract2
# condition can be an expression of Observations that link to Market Observables or it can be a "Call" decision

######
# Step 2. Create a portfolio of 3 trades: 2 European options and 1 callable trade. Callable trade has single call decision and essentially is a European option
######

def european_option(ticker, expiry, strike, ccy, put_call, notional):
    if put_call == "call" : return Cashflow(np.max(Observation(ticker, expiry) - strike, 0), expiry, ccy, notional)
    return Cashflow(np.max(strike - Observation(ticker, expiry), 0), expiry, ccy, notional)

def european_option_autocall(ticker, expiry, strike, ccy, put_call, notional):
    return Option(Observation(ticker, expiry) - strike, Cashflow(Observation(ticker, expiry), expiry, ccy, notional) - Cashflow(strike, expiry, ccy, notional), None)

def european_option_call(ticker, expiry, strike, ccy, put_call, notional):
    return Option(Observation(Ticker("Call", "CPTY"), expiry), Cashflow(Observation(ticker, expiry), expiry, ccy, notional) - Cashflow(strike, expiry, ccy, notional), None)

def digital_option(ticker, expiry, strike, ccy, put_call, notional):
    if put_call == "call" : return Option(Observation(ticker, expiry) - strike, Cashflow(1.0, expiry, ccy, notional), None)
    return Option(strike - Observation(ticker, expiry), Cashflow(1.0, expiry, ccy, notional), None)

def load_example_portfolio():
    trades = []
    trades.append(european_option(Ticker("SPX", "INDEX"), datetime(2025, 1, 1), 100.0, "USD", "call", 1.0))

    # Following 3 contracts should give the same or close result
    trades.append(european_option(Ticker("SPX", "INDEX"), datetime(2035, 1, 1), 100.0, "USD", "call", 1.0)) 
    trades.append(european_option_call(Ticker("SPX", "INDEX"), datetime(2035, 1, 1), 100.0, "USD", "call", 1.0))
    trades.append(european_option_autocall(Ticker("SPX", "INDEX"), datetime(2035, 1, 1), 100.0, "USD", "call", 1.0))

    trades.append(digital_option(Ticker("SPX", "INDEX"), datetime(2035, 1, 1), 100.0, "USD", "call", 1.0))

    return trades

######
# Step 3. Define a function to fuzzify contracts. This function removes all autocall decisions from the portfolio and keeps only call decisions
######

# Remove all Options that are autocallables(i.e. not a Call decisions). Only keep Options that have call decisions
# Here idea is to apply accumulated condition to Cashflows and effectively remove Options.
# A special operation iIF is used to pay Cashflow only if accumulated condition is positive.
# iIF can be implemented as binary opration, or it can be implemented as smooth function
# such as tanh or sigmoid. Effectively defining smooth transition between Contract legs.
# This is mandatory for MonteCarlo path-wise adjoint differentiation

def fuzzify_contract(contract, condition = None, call_ticker = "Call", top_level = True):
    contract = copy.deepcopy(contract, {})
    if isinstance(contract, Cashflow):
        if condition is None:
            return contract
        contract.observable = ObsOp(dependencies=[condition , contract.observable], operation='*') # i.e. we multiply sigmoid by value
        return contract
    if isinstance(contract, Leg):
        return Leg([fuzzify_contract(cf, condition, call_ticker, False) for cf in contract.contracts])
    if isinstance(contract, Option):
        if isinstance(contract.condition, Observation):
            if contract.condition.ticker.symbol == call_ticker:
                return Option(contract.condition, fuzzify_contract(contract.contract1, condition, call_ticker, False), fuzzify_contract(contract.contract2, condition, call_ticker, False))
        if condition is None:
            this_condition1 = ObsOp(dependencies=[contract.condition], operation='sigmoid')
            this_condition2 = (this_condition1 * (-1.0) + (1.0))
        else:
            this_condition1 = ObsOp(dependencies=[condition , ObsOp(dependencies=[contract.condition], operation='sigmoid')], operation='*')
            this_condition2 = (this_condition1 * (-1.0) + (1.0))
        return fuzzify_contract(contract.contract1, this_condition1, call_ticker, False) + fuzzify_contract(contract.contract2, this_condition2, call_ticker, False)
    return contract

# With such definitions we can inspect the portfolio and collect all required observations

def tickers_in_observable(observable):
    if isinstance(observable, Observation):
        return {observable}
    if hasattr(observable, 'dependencies'):
        tickers = set()
        for dep in observable.dependencies:
            tickers.update(tickers_in_observable(dep))
        return tickers
    return set()

def required_observations(portfolio):
    tickers = set()
    # check if portfolio is a single trade or a list of trades
    if isinstance(portfolio, Contract):
        portfolio = [portfolio]
    
    for trade in portfolio:
        if isinstance(trade, Cashflow):
            tickers.update(tickers_in_observable(trade.observable))
        elif isinstance(trade, Option):
            tickers.update(required_observations([trade.contract1, trade.contract2]))
            tickers.update(tickers_in_observable(trade.condition))
        elif isinstance(trade, Leg):
            tickers.update(required_observations(trade.contracts))
    return tickers

if __name__  == "__main__":
    portfolio = load_example_portfolio()

    print(portfolio)
    print(required_observations(portfolio))



if __name__  == "__main__":

    today = datetime(2024, 7, 11)

    # xva_time_steps used for XVA calculations and t0 pricing
    xva_time_steps = [today + relativedelta(days=7 * i) for i in range(52 * 5)]
    last_week = xva_time_steps[-1]
    xva_time_steps += [last_week + relativedelta(days=30 * i) for i in range(12 * 50)]

    call_dates = [obs.fixing_datetime for obs in required_observations(portfolio) if isinstance(obs, Observation) and obs.ticker.symbol == "Call"]

    obs_dates = [obs.fixing_datetime for obs in required_observations(portfolio) if isinstance(obs, Observation) and not obs.ticker.symbol == "Call"]

    xva_time_steps = sorted(set(xva_time_steps + obs_dates + call_dates))

    #print(xva_time_steps)

    #plt.plot(xva_time_steps)
    #plt.show()

######    
# Step 5. Define a function to update observables in contracts. This function is used to update observables in contracts with new market data (historical or simulated)
######

# Note that "cached" values are stored in Contract Observable objects. So variables optimally recalculated on new market data updates. I.e. for step ti to ti+1, we
# can evaluate expressions that can be fully determined at ti+1. 


def get_value(obs, N):  # N is annoying here. use it for np.max() # Error - The requested array has an inhomogeneous shape after 1 dimensions
    if isinstance(obs, float) or isinstance(obs, int):
        return np.full((N), obs)
    return obs.value

def update_observables(contract, market):
    if isinstance(contract, Cashflow):
        return update_observables(contract.observable, market)
    if isinstance(contract, Leg):
        args = True
        for cf in contract.contracts:
            args = args & update_observables(cf, market)
        return args
    if isinstance(contract, Option):
        args = True
        if not (isinstance(contract.condition, Observation) and contract.condition.ticker.symbol == "Call"):
            args = args and update_observables(contract.condition, market)
        args = args and update_observables(contract.contract1, market)
        args = args and update_observables(contract.contract2, market)

        return args

    if hasattr(contract, 'value'):
        return True

    if isinstance(contract, ObsOp):
        # if has value attribute, return

        # update dependencies
        args = True
        for dep in contract.dependencies:
            args = args & update_observables(dep, market)
        if args:
            if (contract.operation == "-"):
                contract.value = get_value(contract.dependencies[0], market.N) - get_value(contract.dependencies[1], market.N)
            elif (contract.operation == "+"):
                contract.value = get_value(contract.dependencies[0], market.N) + get_value(contract.dependencies[1], market.N)
            elif (contract.operation == "&"):
                # TODO:
                contract.value = get_value(contract.dependencies[0], market.N) * get_value(contract.dependencies[1], market.N)
            elif (contract.operation == "/"):
                contract.value = get_value(contract.dependencies[0], market.N) / get_value(contract.dependencies[1], market.N)
            elif (contract.operation == "*"):
                contract.value = get_value(contract.dependencies[0], market.N) * get_value(contract.dependencies[1], market.N)
#            elif (contract.operation == "iIF"):
#                d = [get_value(dep, market.N) for dep in contract.dependencies]
#                contract.value = np.where(d[0] > 0, d[1], 0)  # iIF operation implemented as binary operation. I.e. no smoothing ATM
            elif (contract.operation == "sigmoid"):
                #d = [get_value(dep, market.N) for dep in contract.dependencies]
                #contract.value = 1.0 / (1.0 + np.exp(-d[0]))
                d0 = get_value(contract.dependencies[0], market.N)
                contract.value = np.where(d0 > 0, 1.0, 0)  # iIF operation implemented as binary operation. I.e. no smoothing ATM
            elif (contract.operation == "max"):
                d = [get_value(dep, market.N) for dep in contract.dependencies]
                contract.value = np.max(d, 0)
            else:
                raise NotImplementedError("Operation {} not implemented".format(contract.operation))
        return args

    if isinstance(contract, Observation):
        if contract.fixing_datetime <= market.t:
            contract.value = market.getObservation(contract.ticker, contract.fixing_datetime)
            return True
        return False

    # check if contract is a float constant or None
    if isinstance(contract, float) or isinstance(contract, int) or contract is None:
        return True
    
    return False 

def collect_cashflows(contract, from_t, to_t):
    if isinstance(contract, Cashflow):
        if contract.payment_date > from_t and contract.payment_date <= to_t:
            return [contract]
        return []
    if isinstance(contract, Option):
        # All trades should be fuzzified at this point
        raise NotImplementedError("Options can not be resolved at this point")
    if isinstance(contract, Leg):
        cfs = []
        for cf in contract.contracts:
            cfs += collect_cashflows(cf, from_t, to_t)
        return cfs

    return []

def trade_simulation_dates(contract, pricing_time_steps):
    req_obs = required_observations(contract)

#    for obs in req_obs:
#        if obs.ticker.symbol != "SPX":
#            raise NotImplementedError("Only SPX ticker is supported at this point")

    obs_dates = [obs.fixing_datetime for obs in req_obs]
    if not obs_dates:
        return pricing_time_steps
    
    max_obs_date = max(obs_dates)

    trade_pricing_time_steps = sorted(set(pricing_time_steps + obs_dates))

    # remove all observations that are after max_obs_date
    trade_pricing_time_steps = [t for t in trade_pricing_time_steps if t <= max_obs_date]

    return trade_pricing_time_steps

def price_mc_trade(contract, market_t0, model):
    # Price simple trade without Options. I.e. if autocallable trade should be fuzzified first
    market_prev = market_t0

    clear_values(contract)
    N = model.num_paths

    trade_cf_sum = 0

    vols = model.vols

    np.random.seed(model.seed)

    trade_pricing_time_steps = trade_simulation_dates(contract, model.pricing_time_steps)

    for time_i in range(len(trade_pricing_time_steps)-1):
        shocks = {"SPX": np.random.normal(0, 1.0, N), "USD": 0, "EUR": np.random.normal(0, 1.0, N)}
        next_market = SimulatedMarket(market_prev, shocks, trade_pricing_time_steps[time_i + 1], vols)

        next_market.N = N

        update_observables(contract, next_market)
        cfs = collect_cashflows(contract, trade_pricing_time_steps[time_i], trade_pricing_time_steps[time_i + 1])

        if cfs:
#            print(cfs)
            for cf in cfs:
                this_value = cf.notional * cf.observable.value
                trade_cf_sum += this_value
        market_prev = next_market

    clear_values(contract)        

    return np.average(trade_cf_sum)

# Need Market(model) to check if CashFlow can be priced analytically or needs regression

# returns pair of [PV, sum of cashflows]
# PV - analytical present value of cashflows
# sum - sum of cashflows for regression

def cf_pv_and_sum(contract, t):
    # Use analytical functions where possible
    # Use LS regression where necessary

    cf_pv = 0
    cf_sum = 0

    if isinstance(contract, Cashflow):
        if contract.payment_date >= t:
            if hasattr(contract.observable, 'value'):
                return [0, contract.notional * contract.observable.value]
            return [contract.notional * contract.observable, 0]  # observable is a constant

    if isinstance(contract, Leg):
        for cf in contract.contracts:
            pv_and_sum = cf_pv_and_sum(cf, t)
            cf_pv += pv_and_sum[0]
            cf_sum += pv_and_sum[1]
        return [cf_pv, cf_sum]
    
    # Here we assume that Options are already processed by LS and cont_value is set
    if isinstance(contract, Option):
        return [0, contract.cont_value]

    return [0, 0]

# Function assumes that all cashflows are computed
def find_optimal_calls(contract, regression_vars, xva_time_steps):
    if isinstance(contract, Cashflow):
        return
    if isinstance(contract, Leg):
        for cf in contract.contracts:
            find_optimal_calls(cf, regression_vars, xva_time_steps)
        # Now Leg can only contain regular cashflows
        
        return
    if isinstance(contract, Option):
        if not isinstance(contract.condition, Observation):
            if not contract.condition.ticker.symbol == "Call":
                raise NotImplementedError("Only call decisions are supported at this point")
            
        find_optimal_calls(contract.contract1, regression_vars, xva_time_steps)
        find_optimal_calls(contract.contract2, regression_vars, xva_time_steps)

        call_date = contract.condition.fixing_datetime
        # both contracts are simple list of cashflows
        # value - analytcal value of cash flows
        # sum - sum of cash flows to do LS pricing
        contract1_discounted_cf_value, contract1_dis_cf_sum = cf_pv_and_sum(contract.contract1, call_date)
        contract2_discounted_cf_value, contract2_dis_cf_sum = cf_pv_and_sum(contract.contract2, call_date)

        this_date_regression_vars = {id: regression_vars[id][xva_time_steps.index(call_date)] for id in regression_vars}

        is_contract1_needs_regression = not (isinstance(contract1_dis_cf_sum, float) or isinstance(contract1_dis_cf_sum, int))
        is_contract2_needs_regression = not (isinstance(contract2_dis_cf_sum, float) or isinstance(contract2_dis_cf_sum, int))

        cont_val_regr1m2 = 0
        cont_val_obs = 0

        if is_contract1_needs_regression or is_contract2_needs_regression:
            X = np.array([this_date_regression_vars[reg_id] for reg_id in regression_vars]).T
            regression_vars_ids = [reg_id for reg_id in regression_vars]
            model = LinearRegression()
            model.fit(X, contract1_dis_cf_sum - contract2_dis_cf_sum)
            cont_val_regr1m2 = model.predict(X)
            cont_val_obs = model.intercept_
            for i in range(len(regression_vars)):
                cont_val_obs = model.coef_[i] * Observation(Ticker(regression_vars_ids[i], "SimulatedMarket"), call_date) + cont_val_obs

        reg_condition = contract1_discounted_cf_value + cont_val_regr1m2 > contract2_discounted_cf_value
        this_cont_value = np.where(
            reg_condition
            , contract1_dis_cf_sum + contract1_discounted_cf_value
            , contract2_dis_cf_sum + contract2_discounted_cf_value
        )

        # TODO: contract[1/2]_discounted_cf_value assumed to be constant value here, but should be obs expression

        contract.condition = (cont_val_obs + contract1_discounted_cf_value) - contract2_discounted_cf_value

        contract.cont_value = this_cont_value   # NOTE: We update Option in the contract, so cf_pv_and_sum() can use it
        # TODO: change Option() to fuzzify contract?
        # Create observable for Option condition


if __name__  == "__main__":

    index_values = {"SPX": 100.0, "USD": 1.0, "EUR": 1.2}

    market_t0 = Market(index_values, today)

    N = 1000

    market_prev = market_t0

    print(portfolio[0])
    for trade_i in range(len(portfolio)):
        portfolio[trade_i] = fuzzify_contract(portfolio[trade_i])

    print(portfolio[0])
    #regression_var_ids = ["SPX", "EUR", "BASIS2_SPX", "BASIS3_SPX", "BASIS4_SPX"]
    regression_var_ids = ["SPX", "EUR", "CHEB2_SPX", "CHEB3_SPX", "CHEB4_SPX"]


    if call_dates:
        print("Run LS pass to estimate call decisions. Call Dates: ", call_dates)
        regression_vars = {id: {} for id in regression_var_ids}

        for time_i in range(len(xva_time_steps) - 1):
            shocks = {"SPX": np.random.normal(0, 0.1, N), "USD": 0, "EUR": np.random.normal(0, 0.1, N)}
            market_t1 = SimulatedMarket(market_prev, shocks, xva_time_steps[time_i + 1])

        #    print(spx_path[-1])

            market_t1.N = N
            
            for trade_i in range(len(portfolio)):
                update_observables(portfolio[trade_i], market_t1)        
                # aadc recording ON  # ticker - inputs
        #        cfs = collect_cashflows(portfolio[trade_i], xva_time_steps[time_i], xva_time_steps[time_i + 1])
        #        if cfs:
        #            print(cfs)
        #            portfolio_cfs[trade_i][time_i] = cfs
                # aadc recording off # cashflows - outputs

            if time_i > 0:
                # Regression variables for call decisions
                if xva_time_steps[time_i] in call_dates:
                    for reg_id in regression_var_ids:
                        regression_vars[reg_id][time_i] = market_prev.getObservation(Ticker(reg_id, "INDEX"), xva_time_steps[time_i])

            market_prev = market_t1

        # reverse pass to estimate call decisions
        for trade_i in range(len(portfolio)):
            find_optimal_calls(portfolio[trade_i], regression_vars, xva_time_steps)

        # Fuzzify contracts after call decisions
        for trade_i in range(len(portfolio)):
            portfolio[trade_i] = fuzzify_contract(portfolio[trade_i])


def fuzzify_callable_contract(contract, market_t0, model):

    clear_values(contract)

    contract = fuzzify_contract(contract)
    
    market_prev = market_t0
    regression_var_ids = ["SPX", "EUR", "CHEB2_SPX", "CHEB3_SPX", "CHEB4_SPX"]

    call_dates = [obs.fixing_datetime for obs in required_observations(contract) if isinstance(obs, Observation) and obs.ticker.symbol == "Call"]

    if call_dates is None:
        return contract

#    print(call_dates)

    regression_vars = {id: {} for id in regression_var_ids}

    trade_pricing_time_steps = trade_simulation_dates(contract, model.pricing_time_steps)

    vols = model.vols

#    print(trade_pricing_time_steps)

    np.random.seed(model.seed_ls)

    for time_i in range(len(trade_pricing_time_steps) - 1):
        shocks = {"SPX": np.random.normal(0, 1.0, model.num_paths_ls), "USD": 0, "EUR": np.random.normal(0, 1.0, model.num_paths_ls)}
        next_market = SimulatedMarket(market_prev, shocks, trade_pricing_time_steps[time_i + 1], vols)

        next_market.N = model.num_paths_ls
        
        update_observables(contract, next_market)

#        if time_i > 0:
            # Regression variables for call decisions
        if trade_pricing_time_steps[time_i+1] in call_dates:
            for reg_id in regression_var_ids:
                regression_vars[reg_id][time_i+1] = next_market.getObservation(Ticker(reg_id, "INDEX"), trade_pricing_time_steps[time_i+1])

        market_prev = next_market

    # reverse pass to estimate call decisions
    find_optimal_calls(contract, regression_vars, trade_pricing_time_steps)

    # Fuzzify contracts after call decisions
    contract = fuzzify_contract(contract)

    clear_values(contract)

    return contract


def clear_values(contract):
    if hasattr(contract, 'value'):
        del contract.value
    if isinstance(contract, Cashflow):
        clear_values(contract.observable)
    if isinstance(contract, Leg):
        for cf in contract.contracts:
            clear_values(cf)
    if isinstance(contract, Option):
        if hasattr(contract, 'cont_value'):
            del contract.cont_value
        clear_values(contract.condition), clear_values(contract.contract1), clear_values(contract.contract2)
    if isinstance(contract, ObsOp):
        for dep in contract.dependencies:
            clear_values(dep)

# Price using path-wise MC loop
if __name__  == "__main__":

    market_prev = market_t0
    N = 3000 # different number of MC paths for t0 pricing

    for trade_i in range(len(portfolio)):
        clear_values(portfolio[trade_i])

    trade_cf_sum = [None for _ in range(len(portfolio))]

    for time_i in range(len(xva_time_steps) - 1):
        shocks = {"SPX": np.random.normal(0, 0.1, N), "USD": 0, "EUR": np.random.normal(0, 0.1, N)}
        market_t1 = SimulatedMarket(market_prev, shocks, xva_time_steps[time_i + 1])

        market_t1.N = N
        
        for trade_i in range(len(portfolio)):
            update_observables(portfolio[trade_i], market_t1)        
            # aadc recording ON  # ticker - inputs
            cfs = collect_cashflows(portfolio[trade_i], xva_time_steps[time_i], xva_time_steps[time_i + 1])
            if cfs:
                print(cfs)
                for cf in cfs:
                    this_value = cf.notional * cf.observable.value
                    if trade_cf_sum[trade_i] is None:
                        trade_cf_sum[trade_i] = np.zeros(N)
                    trade_cf_sum[trade_i] += this_value
            # aadc recording off # cashflows - outputs

        market_prev = market_t1

    for trade_i in range(len(portfolio)):
        print("Trade ", trade_i, " price ", np.average(trade_cf_sum[trade_i]))

    # init set to collect trade cfs
    portfolio_cfs = [{} for _ in range(len(portfolio))]

    # Run second Monte Carlo pass for XVA
    market_prev = market_t0

    N = 2000 # different number of MC paths

    regression_vars = {id: {} for id in regression_var_ids}

    spx_path = []

    for trade_i in range(len(portfolio)):
        clear_values(portfolio[trade_i])

    for time_i in range(len(xva_time_steps) - 1):
        shocks = {"SPX": np.random.normal(0, 0.1, N), "USD": 0, "EUR": np.random.normal(0, 0.1, N)}
        market_t1 = SimulatedMarket(market_prev, shocks, xva_time_steps[time_i + 1])

        spx_path.append(market_t1.getObservation(Ticker("SPX", "INDEX"), xva_time_steps[time_i + 1]))
    #    print(spx_path[-1])
        
        market_t1.N = N
        
        for trade_i in range(len(portfolio)):
            update_observables(portfolio[trade_i], market_t1)        
            # aadc recording ON  # ticker - inputs
            cfs = collect_cashflows(portfolio[trade_i], xva_time_steps[time_i], xva_time_steps[time_i + 1])
            if cfs:
                print(cfs)
                portfolio_cfs[trade_i][time_i] = cfs
            # aadc recording off # cashflows - outputs

        if time_i > 0:
            for reg_id in regression_var_ids:
                regression_vars[reg_id][time_i] = market_prev.getObservation(Ticker(reg_id, "INDEX"), xva_time_steps[time_i])

        market_prev = market_t1


    # reverse time LS pass to estimate regression coefficients
    trade_cf_sum = [None for _ in range(len(portfolio))]

    #plt.ion()

    for time_i in reversed(range(len(xva_time_steps) - 1)):
        for trade_i in range(len(portfolio)):
            if time_i in portfolio_cfs[trade_i]:
                for cf in portfolio_cfs[trade_i][time_i]:
                    # TODO: discounting andd payment currency conversion
                    this_value = cf.notional * cf.observable.value
                    if trade_cf_sum[trade_i] is None:
                        trade_cf_sum[trade_i] = this_value
                    else:
                        trade_cf_sum[trade_i] += this_value

            # regress cf_sum on regression_vars
            if (not trade_cf_sum[trade_i] is None) and time_i > 0:
    #            print(time_i, " ", trade_cf_sum[trade_i])
                
                model = LinearRegression()
                features_for_trade = regression_var_ids # TODO: limit to vars need for the trade
                X = np.array([regression_vars[reg_id][time_i] for reg_id in features_for_trade]).T
                model.fit(X, trade_cf_sum[trade_i])

    #            print(time_i, " ", trade_i, " ", model.coef_, " ", model.intercept_)

                biased_price = model.predict(X)
                if (time_i < 10 | time_i > 320):
                    plt.scatter(X[:, 0], biased_price)
                    plt.scatter(X[:, 0], trade_cf_sum[trade_i], s=1, alpha=0.5)
                    plt.show()
                    #clear plot
                    plt.clf()

    for trade_i in range(len(portfolio)):
        print("Trade ", trade_i, " price ", np.average(trade_cf_sum[trade_i]))

        # Price portfolio on xva dates
        # Allocate cashflows to xva dates (optional)
        # Fixings for observables interpolate linearly

        # xva dates without CFs in between (xva dates with variable steps) 30y exp ~ 120 steps
        # Create fmtm daily from t_i-1 t_i use brownian bridge. 
        # Volatility from simulation (creates additional dependency for adjoint pass)
        # cfs - cashflows
        # pass 2 - fmtm backward regression ?? fmtm on portfolio or trade level?
        # pass 3. Collateral - daily time steps + BB
        # pass 4. CVA/DVA

        # Process collateral position based on daily fmtms

        # CVA/DVA ? FVA






    #plt.plot(xva_time_steps[1:], spx_path)
    #plt.show()

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
