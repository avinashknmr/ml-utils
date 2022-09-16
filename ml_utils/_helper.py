def months_diff(start_dt, end_dt, add_one=False):
    """Calculate months difference between 2 dates. To add one extra use parameter add_one=True"""
    a = 1 if add_one else 0
    return (12*(start_dt.year - end_dt.year) + (start_dt.month - end_dt.month) + a)