from time import time
from analytics import *
import aadc


def price_mc_trade_aadc(contract, market_t0_, model):
    # Price simple trade without Options. I.e. if autocallable trade should be fuzzified first

    aadc_func = aadc.Functions()

    market_t0 = copy.deepcopy(market_t0_)

    aadc_func.start_recording()

    market_t0.indexes["SPX"] = aadc.idouble(market_t0.indexes["SPX"])

    indexes_args = {}
    indexes_args["SPX"] = market_t0.indexes["SPX"].mark_as_input()

    market_prev = market_t0

    clear_values(contract)
    N = 1

    trade_cf_sum = 0

    vols = model.vols

    np.random.seed(model.seed)

    trade_pricing_time_steps = trade_simulation_dates(contract, model.pricing_time_steps)

    normals_args = []
    for time_i in range(len(trade_pricing_time_steps)-1):
#        shocks = {"SPX": np.random.normal(0, 1.0, N), "USD": 0, "EUR": np.random.normal(0, 1.0, N)}

        shocks = {"SPX": aadc.idouble(np.random.normal(0, 1.0)), "EUR": aadc.idouble(np.random.normal(0, 1.0))}

        for key, value in shocks.items():
            normals_args.append(value.mark_as_input())
        

        next_market = SimulatedMarket(market_prev, shocks, trade_pricing_time_steps[time_i + 1], vols)

        next_market.N = N

        update_observables(contract, next_market)
        cfs = sum_cashflows(contract, trade_pricing_time_steps[time_i], trade_pricing_time_steps[time_i + 1])

        trade_cf_sum += cfs

        market_prev = next_market

    trade_cf_sum_res = trade_cf_sum[0].mark_as_output()

    clear_values(contract)

    aadc_func.stop_recording()

#    aadc_func.print_passive_extract_locations()

    inputs = {}

    inputs[indexes_args["SPX"]] = market_t0.indexes["SPX"].val()

    np.random.seed(model.seed)

    # measure rn gen time

    start_time = time()
    for i in range(len(normals_args)):
        inputs[normals_args[i]] = np.random.normal(0, 1.0, model.num_paths)
    end_time = time()
#    print("Random number generation time = ", end_time - start_time)

    request = {trade_cf_sum_res: [indexes_args["SPX"]]};

    workers = aadc.ThreadPool(4)
    res = aadc.evaluate(aadc_func, request, inputs, workers)

#    print("AADC Price = ", np.average(res[0][trade_cf_sum_res]))

#    print("AADC dPrice/dSPX = ", np.average(res[1][trade_cf_sum_res][indexes_args["SPX"]]))         

    out = {}
    out["price"] = np.average(np.average(res[0][trade_cf_sum_res]))
    out["delta"] = np.average(np.average(res[1][trade_cf_sum_res][indexes_args["SPX"]]))

    return out
