# author:"flt"
# data:10/17/2024 4:52 PM
from dowhy.datasets import linear_dataset
from dowhy import CausalModel
import econml

# Config dict to set the logging level
import logging.config
DEFAULT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'loggers': {
        '': {
            'level': 'WARN',
        },
    }
}

logging.config.dictConfig(DEFAULT_LOGGING)
# Disabling warnings output
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

inspect_datasets = True
inspect_models = True
inspect_identified_estimands = True
inspect_estimates = True
inspect_refutations = True

