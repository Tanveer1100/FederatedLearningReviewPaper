
import time
import psutil

class ComputeLogger:
    def __init__(self):
        self.start_time = None
        self.cpu_before = None

    def start(self):
        self.start_time = time.time()
        self.cpu_before = psutil.cpu_percent(interval=None)

    def stop(self):
        duration = time.time() - self.start_time
        memory = psutil.virtual_memory().used / (1024 ** 2)  # MB
        cpu_after = psutil.cpu_percent(interval=None)
        return {
            'duration_sec': duration,
            'memory_MB': memory,
            'cpu_load': cpu_after - self.cpu_before
        }
