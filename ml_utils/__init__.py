import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

__version__ = '0.1.0'

__pdoc__ = {}
__pdoc__['ml_utils.measure.Metrics'] = False
__pdoc__['ml_utils.draw.trend_line'] = False