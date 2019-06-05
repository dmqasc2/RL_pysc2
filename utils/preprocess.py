import numpy as np


class preprocess:
    def __init__(self, ):
        self.screen_channels = len(features.SCREEN_FEATURES)
        self.minimap_channels = len(features.MINIMAP_FEATURES)
        self.flat_channels = len(flat_specs_sv)
        self.available_actions_channels = NUM_FUNCTIONS