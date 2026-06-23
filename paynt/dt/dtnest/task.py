
from paynt.dt.task import DtTask

class DtNestTask(DtTask):

    def __init__(self, properties, error_threshold, tree_depth=7, initial_tree=None, timeout=600):

        super().__init__(properties, tree_depth, timeout=timeout)
        self.error_threshold = error_threshold
        self.initial_tree = initial_tree
