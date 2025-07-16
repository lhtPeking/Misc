from multiprocessing import Process
import time

class GlobalTimerModule(Process):
    def __init__(self, shared):
        Process.__init__(self)

        self.shared = shared

    def run(self):

        while self.shared.running.value == 1:
            self.shared.global_timer.value = time.perf_counter()
            #time.sleep(0.001)