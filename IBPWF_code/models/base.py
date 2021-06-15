from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

class Base:

    def __init__(self, config):
        self.config = config
        tf.reset_default_graph()


    @abstractmethod
    def forward(self, x, task_id):
        pass

    @abstractmethod
    def register_task(self):
        pass


