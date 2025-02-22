import QuantLib as ql
from typing import Dict, List, Optional, Union
from datetime import datetime
from contract import Cashflow, Leg, Option, Observation, Contract, Ticker
import numpy as np

# Module-level utility functions for QuantLib conversions
def to_ql_date(date_input: Union[datetime, str]) -> ql.Date:
    """Convert datetime or string to QuantLib Date"""
    if isinstance(date_input, str):
        date_input = datetime.strptime(date_input, '%Y-%m-%d')
    return ql.Date(date_input.day, date_input.month, date_input.year)

def to_datetime(ql_date: ql.Date) -> datetime:
    """Convert QuantLib Date to Python datetime"""
    return datetime(ql_date.year(), ql_date.month(), ql_date.dayOfMonth())

def parse_frequency(freq_str: str) -> ql.Period:
    """Convert frequency string to QuantLib Period"""
    number = int(freq_str[:-1])
    unit = freq_str[-1].upper()
    unit_map = {'D': ql.Days, 'W': ql.Weeks, 'M': ql.Months, 'Y': ql.Years}
    return ql.Period(number, unit_map[unit])

def get_calendar(cal_name: str) -> ql.Calendar:
    """Convert calendar name to QuantLib Calendar"""
    calendar_map = {
        'USA': ql.UnitedStates(ql.UnitedStates.GovernmentBond),
        'UK': ql.UnitedKingdom(),
        'TARGET': ql.TARGET(),
        'JPY': ql.Japan()
    }
    return calendar_map.get(cal_name.upper(), ql.UnitedStates(ql.UnitedStates.GovernmentBond))

def get_business_day_convention(convention: str) -> int:
    """Convert business day convention string to QuantLib enum"""
    convention_map = {
        'F': ql.Following,
        'MF': ql.ModifiedFollowing,
        'P': ql.Preceding,
        'MP': ql.ModifiedPreceding
    }
    return convention_map.get(convention.upper(), ql.ModifiedFollowing)

def get_date_generation(gen_rule: str) -> int:
    """Convert date generation string to QuantLib enum"""
    generation_map = {
        'backward': ql.DateGeneration.Backward,
        'forward': ql.DateGeneration.Forward,
        'zero': ql.DateGeneration.Zero
    }
    return generation_map.get(gen_rule.lower(), ql.DateGeneration.Forward)

def get_day_counter(day_count_str: str) -> ql.DayCounter:
    """Convert day count string to QuantLib DayCounter"""
    day_count_map = {
        '30/360': ql.Thirty360(ql.Thirty360.BondBasis),
        'ACT/360': ql.Actual360(),
        'ACT/365': ql.Actual365Fixed(),
        'ACT/ACT': ql.ActualActual(ql.ActualActual.ISDA),
        'ACT/ACT ISDA': ql.ActualActual(ql.ActualActual.ISDA),
        'ACT/ACT ISMA': ql.ActualActual(ql.ActualActual.ISMA),
        'ACT/ACT AFB': ql.ActualActual(ql.ActualActual.AFB)
    }
    return day_count_map.get(day_count_str.upper(), ql.Thirty360(ql.Thirty360.BondBasis))

def format_date(ql_date: ql.Date) -> str:
    """Convert QuantLib date to string format YYYY-MM-DD"""
    return f"{ql_date.year()}-{ql_date.month():02d}-{ql_date.dayOfMonth():02d}"

def fixed_leg(
    effective_date: Union[datetime, str],
    termination_date: Union[datetime, str],
    fixed_rate: float,
    notional: float,
    currency: str,
    payment_frequency: str = "6M",
    calendar_name: str = "USA",
    business_day_convention: str = "MF",
    date_generation: str = "forward",
    end_of_month: bool = False,
    day_count: str = "30/360"
) -> List[Dict[str, any]]:
    """
    Calculate fixed leg payment dates and cash flows.
    
    Args:
        effective_date: Start date as datetime or string 'YYYY-MM-DD'
        termination_date: End date as datetime or string 'YYYY-MM-DD'
        fixed_rate: Annual fixed rate as decimal (e.g., 0.03 for 3%)
        notional: Notional amount
        currency: Currency code (e.g., 'USD', 'EUR')
        payment_frequency: String frequency specifier ('1M', '3M', '6M', '1Y', etc.)
        calendar_name: Calendar name ('USA', 'UK', 'TARGET', etc.)
        business_day_convention: Date convention ('MF'=Modified Following, 'F'=Following, 'P'=Preceding)
        date_generation: Schedule generation rule ('forward', 'backward')
        end_of_month: Boolean flag for end of month rule
        day_count: Day count convention ('30/360', 'ACT/360', 'ACT/365', etc.)
        
    Returns:
        List of dictionaries containing payment information for each period
    """
    # Convert inputs to QuantLib types
    ql_effective_date = to_ql_date(effective_date)
    ql_termination_date = to_ql_date(termination_date)
    ql_frequency = parse_frequency(payment_frequency)
    ql_calendar = get_calendar(calendar_name)
    ql_convention = get_business_day_convention(business_day_convention)
    ql_date_generation = get_date_generation(date_generation)
    ql_day_counter = get_day_counter(day_count)
    
    # Create the schedule
    schedule = ql.Schedule(
        ql_effective_date,
        ql_termination_date,
        ql_frequency,
        ql_calendar,
        ql_convention,
        ql_convention,
        ql_date_generation,
        end_of_month
    )
    
    # Initialize the results list
    fixed_leg_flows = []
    
    # Calculate cash flows for each period
    for i in range(len(schedule)-1):
        start_date = schedule[i]
        end_date = schedule[i+1]
        payment_date = ql_calendar.adjust(end_date, ql_convention)
        
        # Calculate day count fraction
        year_fraction = ql_day_counter.yearFraction(start_date, end_date)
        
        # Calculate fixed payment
        payment_amount = fixed_rate * year_fraction

        cf = Cashflow(payment_amount, to_datetime(payment_date), currency, notional)

        fixed_leg_flows.append(cf)
    
    return Leg(fixed_leg_flows)


def float_leg(
    effective_date: Union[datetime, str],
    termination_date: Union[datetime, str],
    notional: float,
    currency: str,
    index_id: str,
    payment_frequency: str = "3M",
    calendar_name: str = "USA",
    business_day_convention: str = "MF",
    date_generation: str = "forward",
    end_of_month: bool = False,
    day_count: str = "ACT/360",
    fixing_days: int = 2
) -> Leg:
    """
    Calculate floating leg payment dates and cash flows.
    
    Args:
        effective_date: Start date as datetime or string 'YYYY-MM-DD'
        termination_date: End date as datetime or string 'YYYY-MM-DD'
        notional: Notional amount
        currency: Currency code (e.g., 'USD', 'EUR')
        index_id: Reference rate index identifier
        payment_frequency: String frequency specifier ('1M', '3M', '6M', '1Y', etc.)
        calendar_name: Calendar name ('USA', 'UK', 'TARGET', etc.)
        business_day_convention: Date convention ('MF'=Modified Following, 'F'=Following, 'P'=Preceding)
        date_generation: Schedule generation rule ('forward', 'backward')
        end_of_month: Boolean flag for end of month rule
        day_count: Day count convention ('ACT/360', 'ACT/365', etc.)
        fixing_days: Number of fixing days before start of period
        
    Returns:
        contract.Leg containing the floating rate payments
    """
    # Convert inputs to QuantLib types
    ql_effective_date = to_ql_date(effective_date)
    ql_termination_date = to_ql_date(termination_date)
    ql_frequency = parse_frequency(payment_frequency)
    ql_calendar = get_calendar(calendar_name)
    ql_convention = get_business_day_convention(business_day_convention)
    ql_date_generation = get_date_generation(date_generation)
    ql_day_counter = get_day_counter(day_count)
    
    # Create the schedule
    schedule = ql.Schedule(
        ql_effective_date,
        ql_termination_date,
        ql_frequency,
        ql_calendar,
        ql_convention,
        ql_convention,
        ql_date_generation,
        end_of_month
    )
    
    # Initialize the results list
    floating_leg_flows = []
    
    # Calculate cash flows for each period
    for i in range(len(schedule)-1):
        start_date = schedule[i]
        end_date = schedule[i+1]
        payment_date = ql_calendar.adjust(end_date, ql_convention)
        
        # Calculate fixing date (observation time)
        fixing_date = ql_calendar.advance(start_date, -fixing_days, ql.Days)
        
        # Calculate day count fraction
        year_fraction = ql_day_counter.yearFraction(start_date, end_date)
        
        # Create cashflow with Observation
        cf = Cashflow(
            year_fraction * Observation(index_id, to_datetime(fixing_date)),
            to_datetime(payment_date),
            currency,
            notional
        )
        
        floating_leg_flows.append(cf)
    
    return Leg(floating_leg_flows)


def swap(
    effective_date: Union[datetime, str],
    termination_date: Union[datetime, str],
    notional: float,
    fixed_rate: float,
    currency: str,
    index_id: str,
    fixed_payment_frequency: str = "6M",
    float_payment_frequency: str = "3M",
    calendar_name: str = "USA",
    business_day_convention: str = "MF",
    date_generation: str = "forward",
    end_of_month: bool = False,
    fixed_day_count: str = "30/360",
    float_day_count: str = "ACT/360",
    fixing_days: int = 2
) -> List[Leg]:
    """
    Create an interest rate swap with fixed and floating legs.
    
    Args:
        effective_date: Start date as datetime or string 'YYYY-MM-DD'
        termination_date: End date as datetime or string 'YYYY-MM-DD'
        notional: Notional amount
        fixed_rate: Annual fixed rate as decimal (e.g., 0.03 for 3%)
        currency: Currency code (e.g., 'USD', 'EUR')
        index_id: Reference rate index identifier
        fixed_payment_frequency: Frequency for fixed leg ('6M', '1Y', etc.)
        float_payment_frequency: Frequency for floating leg ('1M', '3M', etc.)
        calendar_name: Calendar name ('USA', 'UK', 'TARGET', etc.)
        business_day_convention: Date convention ('MF'=Modified Following, 'F'=Following)
        date_generation: Schedule generation rule ('forward', 'backward')
        end_of_month: Boolean flag for end of month rule
        fixed_day_count: Day count convention for fixed leg ('30/360', etc.)
        float_day_count: Day count convention for float leg ('ACT/360', etc.)
        fixing_days: Number of fixing days before start of period
        
    Returns:
        List of contract.Leg containing [fixed_leg, float_leg]
    """
    # Generate fixed leg
    fixed = fixed_leg(
        effective_date=effective_date,
        termination_date=termination_date,
        fixed_rate=fixed_rate,
        notional=notional,
        currency=currency,
        payment_frequency=fixed_payment_frequency,
        calendar_name=calendar_name,
        business_day_convention=business_day_convention,
        date_generation=date_generation,
        end_of_month=end_of_month,
        day_count=fixed_day_count
    )
    
    # Generate floating leg
    floating = float_leg(
        effective_date=effective_date,
        termination_date=termination_date,
        notional=-notional,
        currency=currency,
        index_id=index_id,
        payment_frequency=float_payment_frequency,
        calendar_name=calendar_name,
        business_day_convention=business_day_convention,
        date_generation=date_generation,
        end_of_month=end_of_month,
        day_count=float_day_count,
        fixing_days=fixing_days
    )
    
    return Leg([fixed, floating])


def physical_swaption(
    option_expiry: datetime,
    underlying_swap: Contract,
    counterparty: str = "CPTY"
) -> Option:
    """
    Create a physically settled swaption.
    
    Args:
        option_expiry: Expiration date of the option
        underlying_swap: The underlying swap contract
        counterparty: Counterparty identifier (default: "CPTY")
        option_type: Option type, either "Call" or "Put" (default: "Call")
        
    Returns:
        Option contract that delivers the underlying swap if exercised
    """
    return Option(
        Observation(Ticker("Call", counterparty), option_expiry),
        underlying_swap,
        None
    )

def cash_settled_swaption(
    option_expiry: datetime,
    swap_rate_ticker: Ticker,
    notional: float,
    currency: str,
    strike: float,
    payment_date: Optional[datetime] = None,
    option_type: str = "Call"
) -> Cashflow:
    """
    Create a cash-settled swaption.
    
    Args:
        option_expiry: Expiration date of the option
        swap_rate_obs_id: Identifier for the swap rate observation
        notional: Notional amount
        currency: Currency code (e.g., 'USD', 'EUR')
        strike: Strike rate for the option
        payment_date: Settlement date (defaults to option_expiry if None)
        option_type: Option type, either "Call" or "Put" (default: "Call")
        
    Returns:
        Cashflow contract representing the cash settlement amount
    """
    # If payment_date not specified, use option expiry
    settlement_date = payment_date if payment_date is not None else option_expiry
    
    # Create the payoff based on option type
    if option_type.upper() == "CALL":
        payoff = np.max(Observation(swap_rate_ticker, option_expiry) - strike, 0)
    else:  # PUT
        payoff = np.max(strike - Observation(swap_rate_ticker, option_expiry), 0)
    
    return Cashflow(
        payoff,
        settlement_date,
        currency,
        notional
    )


def filter_cashflows(
    contract: Union[Contract, List[Contract]], 
    start_date: datetime, 
    end_date: datetime
) -> Optional[Contract]:
    """
    Filter cash flows within time interval (start_date, end_date].
    
    Args:
        contract: Single contract or list of contracts
        start_date: Start date (exclusive)
        end_date: End date (inclusive)
        
    Returns:
        New contract containing only cash flows within the interval,
        or None if no cash flows remain
    """
    # Handle list of contracts
    if isinstance(contract, list):
        filtered = [filter_cashflows(c, start_date, end_date) for c in contract]
        # Remove None results
        filtered = [c for c in filtered if c is not None]
        if not filtered:
            return None
        if len(filtered) == 1:
            return filtered[0]
        return Leg(filtered)

    # Handle single contract types
    if isinstance(contract, Cashflow):
        if start_date < contract.payment_date <= end_date:
            return contract
        return None
        
    elif isinstance(contract, Leg):
        filtered = filter_cashflows(contract.contracts, start_date, end_date)
        return filtered
        
    elif isinstance(contract, Option):
        # Filter both contract branches
        contract1 = filter_cashflows(contract.contract1, start_date, end_date) if contract.contract1 else None
        contract2 = filter_cashflows(contract.contract2, start_date, end_date) if contract.contract2 else None
        
        # If both branches are None, return None
        if contract1 is None and contract2 is None:
            return None
            
        # Create new Option with filtered contracts
        return Option(contract.condition, contract1, contract2)
    
    return None


# Example usage:
if __name__ == "__main__":
    # Example parameters using native Python types
    fixed_leg_cf = fixed_leg(
        effective_date="2025-02-28",
        termination_date="2030-02-28",
        fixed_rate=0.03,  # 3.00%
        notional=10_000_000,  # 10 million
        currency="USD",
        payment_frequency="3M",
        calendar_name="USA",
        day_count="30/360"
    )
    
    print("Fixed leg:")
    print(fixed_leg_cf)

    leg = float_leg(
        effective_date="2025-02-28",
        termination_date="2030-02-28",
        notional=10_000_000,  # 10 million
        currency="USD",
        index_id="USD-LIBOR-3M",
        payment_frequency="3M",
        calendar_name="USA",
        day_count="ACT/360"
    )

    print("Floating leg:")
    print(leg)

    legs = swap(
        effective_date="2025-02-28",
        termination_date="2030-02-28",
        notional=10_000_000,  # 10 million
        fixed_rate=0.03,  # 3.00%
        currency="USD",
        index_id="USD-LIBOR-3M",
        fixed_payment_frequency="6M",
        float_payment_frequency="3M",
        calendar_name="USA"
    )

    print("Swap legs:")
    print(legs)

    # Create a physically settled swaption
    swaption = physical_swaption(
        option_expiry=datetime(2025, 2, 10),
        underlying_swap=legs
    )

    print("Physically settled swaption:")
    print(swaption)

    # Create a cash-settled swaption
    cash_swaption = cash_settled_swaption(
        option_expiry=datetime(2025, 2, 10),
        swap_rate_ticker="USD-SWAP-10Y",
        notional=10_000_000,
        currency="USD",
        strike=0.03,
        payment_date=datetime(2025, 2, 12),
        option_type="Call"
    )

    print("Cash-settled swaption:")
    print(cash_swaption)

if __name__ == "__main__":
    from datetime import datetime, timedelta
    import contract
    from contract import Observation, Ticker
    
    # Create some test cash flows
    start = datetime(2025, 1, 1)
    dates = [start + timedelta(days=i*90) for i in range(8)]  # 8 quarterly payments
    
    # Create a leg with multiple cash flows
    flows = [
        Cashflow(
            Observation("USD-LIBOR-3M", d),
            d, 
            "USD", 
            1_000_000
        ) for d in dates
    ]
    leg = Leg(flows)
    
    # Create an option structure
    option = Option(
        Observation(Ticker("Call", "CPTY"), dates[2]),
        leg,
        None
    )
    
    # Test filtering
    print("Original structure:")
    print(option)
    print("\nFiltered structure (6 months to 1 year):")
    filtered = filter_cashflows(
        option,
        datetime(2025, 6, 30),
        datetime(2026, 1, 1)
    )
    print(filtered)


def bermudan_cancellable(
    exercise_dates: List[datetime],
    underlying_swap: Contract,
    counterparty: str = "CPTY",
    option_type: str = "Call"
) -> Contract:
    """
    Create a Bermudan cancellable swap structure by building nested options backwards.
    At each exercise date, option is to cancel remaining flows (exercise to None).
    Current contract includes the flows up to next exercise date.
    
    Args:
        exercise_dates: List of exercise dates in ascending order
        underlying_swap: The underlying swap contract
        counterparty: Counterparty identifier
        option_type: Option type ("Call" or "Put")
        
    Returns:
        Contract representing the Bermudan cancellable structure
    """
    if not exercise_dates:
        return underlying_swap
        
    # Ensure dates are sorted in ascending order
    exercise_dates = sorted(exercise_dates)
    
    # Start with the final exercise date
    current_contract = filter_cashflows(
        underlying_swap,
        start_date=exercise_dates[-1],
        end_date=datetime.max
    )
    
    # Loop backwards through exercise dates
    for i in range(len(exercise_dates) - 1, -1, -1):
        exercise_date = exercise_dates[i]
        
        # Get flows for the current period (up to next exercise date or end)
        next_exercise = exercise_dates[i + 1] if i < len(exercise_dates) - 1 else datetime.max
        period_flows = filter_cashflows(
            underlying_swap,
            start_date=exercise_date,
            end_date=next_exercise
        )
        
        # Add period flows to current contract structure
        if current_contract is not None:
            if period_flows is not None:
                current_contract = period_flows + current_contract
        else:
            current_contract = period_flows
            
        # Create option at this exercise date
        current_contract = Option(
            Observation(Ticker(option_type, counterparty), exercise_date),
            None,  # Exercise to cancel (None)
            current_contract  # Continue with existing flows if not exercised
        )
    
    # Add initial flows (before first exercise date)
    initial_flows = filter_cashflows(
        underlying_swap,
        start_date=datetime.min,
        end_date=exercise_dates[0]
    )
    
    if initial_flows is not None:
        current_contract = initial_flows + current_contract
        
    return current_contract


# Example usage:
if __name__ == "__main__":
    from datetime import datetime, timedelta
    
    def to_datetime(ql_date: ql.Date) -> datetime:
        """Convert QuantLib Date to Python datetime"""
        return datetime(ql_date.year(), ql_date.month(), ql_date.dayOfMonth())
    
    # Swap parameters
    start_date = ql.Date(1, 1, 2025)
    end_date = ql.Date(1, 1, 2030)
    payment_freq = ql.Period(ql.Semiannual)
    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    business_convention = ql.ModifiedFollowing
    date_gen = ql.DateGeneration.Forward
    month_end = False
    
    # Create swap schedule
    swap_schedule = ql.Schedule(
        start_date,
        end_date,
        payment_freq,
        calendar,
        business_convention,
        business_convention,
        date_gen,
        month_end
    )
    
    # Create underlying swap
    underlying_swap = swap(
        effective_date=to_datetime(start_date),
        termination_date=to_datetime(end_date),
        notional=10_000_000,
        fixed_rate=0.03,
        currency="USD",
        index_id="USD-LIBOR-3M"
    )
    
    # Create exercise schedule (e.g., starting after 1Y non-call period)
    non_call_period = ql.Period(1, ql.Years)
    exercise_start = calendar.advance(start_date, non_call_period)
    
    # Convert payment dates to exercise dates (after non-call period)
    exercise_dates = [
        to_datetime(date) for date in swap_schedule 
        if date >= exercise_start and date < end_date
    ]
    
    # Create Bermudan cancellable structure
    bermudan = bermudan_cancellable(
        exercise_dates=exercise_dates,
        underlying_swap=underlying_swap,
        counterparty="BANK_A",
        option_type="Call"
    )
    
    print("Bermudan Cancellable Structure:")
    print(bermudan)
    print("\nExercise dates (after 1Y non-call period):")
    for date in exercise_dates:
        print(f"- {date.strftime('%Y-%m-%d')}")
