from dateutil.relativedelta import relativedelta

def adjust_date(date, calendar='TARGET', date_adjustment_convention='mfw'):
    return date

def schedule(start_date, tenor=None, frequency=relativedelta(months=6), calendar='TARGET', date_adjustment_convention='mfw'):
    end_date = start_date + tenor
    result = []
    while end_date > start_date:
        end = adjust_date(end_date, calendar, date_adjustment_convention)
        end_date = end_date - frequency
        start = adjust_date(end_date, calendar, date_adjustment_convention)
        if start < start_date:
            start = start_date

        result = [(start, end)] + result

    return result
