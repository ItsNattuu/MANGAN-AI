from typing import Any


class PipelineResult:

    def __init__(self):

        self.layer1 = []
        self.layer2 = []
        self.layer3 = []
        self.layer4 = {}

    def to_dict(self):

        return {
            "layer1": self.layer1,
            "layer2": self.layer2,
            "layer3": self.layer3,
            "layer4": self.layer4
        }
