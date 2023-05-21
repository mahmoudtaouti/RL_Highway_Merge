from keras.callbacks import TensorBoard
import tensorflow as tf
from datetime import datetime
# by sentdex, reinforcement learning tuorial on youtube. 
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self,log_dir = 'logs/', **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        # Define your log directory
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = log_dir + current_time
        # Create a summary writer
        self.writer = tf.summary.create_file_writer(log_dir)
        

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)
    
    
    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    #def update_stats(self, **stats):
    #    self._write_logs(self._prepare_logs(stats), self.step)

    def _prepare_logs(self, logs):
        if self.model.stop_training:
            return
        return {k: v for k, v in logs.items() if k in self.model.metrics_names}

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()