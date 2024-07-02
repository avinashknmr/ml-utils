import yaml

__version__ = '0.1.0'

__pdoc__ = {}
__pdoc__['ml_utils.measure.Metrics'] = False
__pdoc__['ml_utils.draw.trend_line'] = False

def parse_config(filename):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    return config