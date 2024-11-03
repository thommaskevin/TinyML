class Config:
    """
    BaseMobileNet config object
    """
    def __init__(self):
        """

        """
        self.learning_rate = 0.001
        self.loss = "sparse_categorical_crossentropy"
        self.metrics = ["sparse_categorical_accuracy"]
        self.checkpoint_min_accuracy = 0.7
        self.batch_size = 32
        self.verbosity = 1
        self.checkpoint_path = ""
