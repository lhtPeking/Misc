from multiprocessing import Process
import numpy as np
import os
import importlib
import time
import imageio

import psutil, os

class StimulusModule(Process):
    def __init__(self, shared):
        Process.__init__(self)

        self.shared = shared

    def remove_module(self):

        for fish_index in range(4):
            self.shared.stimulus_flow_control_currently_running[fish_index].value = 0

        if self.stimulus_widget is not None:
            try:
                self.stimulus_widget.destroy()

                del self.stimulus_widget
                del self.module
                self.stimulus_widget = None

            except Exception as e:
                print("Stimulus file error", e)

            self.shared.stimulus_information_name_l.value = 0
            self.shared.stimulus_information_number_of_stimuli.value = 0
            self.shared.stimulus_information_time_per_stimulus.value = 0


    def run(self):


        # set the priority of the scanning library to high (don't really know if this improves things...)
        p = psutil.Process(os.getpid())
        p.nice(psutil.HIGH_PRIORITY_CLASS)
        print(p.nice())

        self.stimulus_widget = None

        # create empty lists for the fish
        self.stimulus_data = [[] for _ in range(4)]
        self.stimulus_information = [[] for _ in range(4)]

        while self.shared.running.value == 1:

            if self.shared.stimulus_configuration_stimulus_path_update_requested.value == 1:
                self.shared.stimulus_configuration_stimulus_path_update_requested.value = 0
                full_path_to_module = bytearray(self.shared.stimulus_configuration_stimulus_path[:self.shared.stimulus_configuration_stimulus_path_l.value]).decode()

                self.remove_module()

                try:

                    module_dir, module_file = os.path.split(full_path_to_module)
                    module_name, module_ext = os.path.splitext(module_file)

                    spec = importlib.util.spec_from_file_location(module_name, full_path_to_module)

                    self.module = spec.loader.load_module()

                    self.stimulus_widget = self.module.MyApp(self.shared)

                    stimulus_name = os.path.basename(full_path_to_module).split(".")[0].encode()

                    self.shared.stimulus_information_name[:len(stimulus_name)] = stimulus_name
                    self.shared.stimulus_information_name_l.value = len(stimulus_name)
                    self.shared.stimulus_information_number_of_stimuli.value = self.stimulus_widget.stimulus_number_of_stimuli
                    self.shared.stimulus_information_time_per_stimulus.value = self.stimulus_widget.stimulus_time_per_stimulus

                    self.last_global_time = self.shared.global_timer.value

                except Exception as e:
                    print("Stimulus file error", e)


            if self.stimulus_widget is not None:

                ########
                # Get the dt
                new_global_time = self.shared.global_timer.value
                dt = new_global_time - self.last_global_time
                self.last_global_time = new_global_time

                for fish_index in range(4):

                    ########
                    # Update the alignment
                    self.stimulus_widget.fish_nodes[fish_index].setPos(self.shared.stimulus_configuration_set_x_position[fish_index].value, 1, self.shared.stimulus_configuration_set_y_position[fish_index].value)
                    self.stimulus_widget.fish_nodes[fish_index].setScale(self.shared.stimulus_configuration_set_scale[fish_index].value, 1, self.shared.stimulus_configuration_set_scale[fish_index].value)
                    self.stimulus_widget.fish_nodes[fish_index].setHpr(0, 0, self.shared.stimulus_configuration_set_rotation[fish_index].value)

                    ########
                    # Data acquisition requested?
                    if self.shared.experiment_flow_control_start_acquire_stimulus_data_requested[fish_index].value == 1:
                        self.shared.experiment_flow_control_start_acquire_stimulus_data_requested[fish_index].value = 0
                        self.shared.experiment_flow_control_currently_acquire_stimulus_data[fish_index].value = 1

                        self.stimulus_data[fish_index] = []
                        self.stimulus_information[fish_index] = []

                    #######
                    # Stimulus termination and next stimulus?
                    if self.shared.stimulus_flow_control_next_stimulus_requested[fish_index].value == 1:
                        self.shared.stimulus_flow_control_next_stimulus_requested[fish_index].value = 0

                        # store the information about the previous stimulus
                        if self.shared.experiment_flow_control_currently_acquire_stimulus_data[fish_index].value == 1:
                            self.stimulus_information[fish_index].append(
                                [self.shared.stimulus_flow_control_start_time[fish_index].value,
                                 self.shared.global_timer.value,  # end time
                                 self.shared.stimulus_flow_control_start_index[fish_index].value,
                                 self.shared.stimulus_flow_control_result_info[fish_index].value])

                        # pick a new random stimulus
                        self.shared.stimulus_flow_control_index[fish_index].value = np.random.randint(self.shared.stimulus_information_number_of_stimuli.value)  # start with a random stimulus
                        self.shared.stimulus_flow_control_start_requested[fish_index].value = 1

                    ########
                    # Stimulus start requested?
                    if self.shared.stimulus_flow_control_start_requested[fish_index].value == 1:
                        self.shared.stimulus_flow_control_start_requested[fish_index].value = 0
                        
                        self.shared.recorded_timeindex[fish_index].value = 0
                        self.shared.stimulus_flow_control_currently_running[fish_index].value = 1
                        self.shared.stimulus_flow_control_next_stimulus_requested[fish_index].value = 0

                        # I added this here and commented it out above...
                        #self.shared.stimulus_flow_control_index[fish_index].value = np.random.randint(self.shared.stimulus_information_number_of_stimuli.value)  # start with a random stimulus

                        # rememember these value as their will be stored later in the stimulus initiation information box
                        self.shared.stimulus_flow_control_start_time[fish_index].value = self.shared.global_timer.value
                        self.shared.stimulus_flow_control_start_index[fish_index].value = self.shared.stimulus_flow_control_index[fish_index].value
                        self.shared.stimulus_flow_control_result_info[fish_index].value = -1 # not specified
                        
                        self.stimulus_widget.init_stimulus(fish_index, self.shared.stimulus_flow_control_index[fish_index].value)


                    ########
                    # Update the stimulus if running, and store stimulus data if storing had been requested
                    if self.shared.stimulus_flow_control_currently_running[fish_index].value == 1:
                        self.shared.stimulus_flow_control_current_time[fish_index].value = self.shared.global_timer.value - self.shared.stimulus_flow_control_start_time[fish_index].value

                        additional_stimulus_data = self.stimulus_widget.update_stimulus(fish_index, self.shared.stimulus_flow_control_index[fish_index].value, self.shared.stimulus_flow_control_current_time[fish_index].value, dt)

                        if additional_stimulus_data is not None:
                            # Convert this into a string for displaying purposes
                            info = ""

                            for val in additional_stimulus_data:
                                info += "{:.2f}; ".format(val)

                            info = info[:-2].encode()

                            # Display that info somewhere, TODO

                            # add the information to the data stream
                            if self.shared.experiment_flow_control_currently_acquire_stimulus_data[fish_index].value == 1:
                                self.stimulus_data[fish_index].append(np.r_[self.shared.global_timer.value, additional_stimulus_data])

                    if self.shared.experiment_flow_control_store_stimulus_data_requested[fish_index].value == 1:
                        self.shared.experiment_flow_control_store_stimulus_data_requested[fish_index].value = 0

                        self.shared.experiment_flow_control_currently_storing_stimulus_data[fish_index].value = 1
                        self.shared.experiment_flow_control_currently_acquire_stimulus_data[fish_index].value = 0

                        # save the tail data from that trial
                        rawdata_path = bytearray(self.shared.experiment_flow_control_rawdata_path[fish_index][:self.shared.experiment_flow_control_rawdata_path_l[fish_index].value]).decode()

                        filename = os.path.join(rawdata_path, "trial{:03d}_stimulus_data".format(self.shared.experiment_flow_control_current_trial.value))

                        stimulus_data = np.array(self.stimulus_data[fish_index])
                        stimulus_information = np.array(self.stimulus_information[fish_index])

                        '''
                        if len(stimulus_data) > 0:
                            np.savez_compressed(filename,
                                stimulus_start_times =stimulus_information[:, 0],
                                stimulus_end_times =stimulus_information[:, 1],
                                stimulus_start_indices=stimulus_information[:, 2].astype(np.uint),
                                stimulus_result_info=stimulus_information[:, 3],
                                stimulus_time=stimulus_data[:, 0],
                                stimulus_data=stimulus_data[:, 1:])
                        else: # always store information about the start times of the visual stimuli
                            np.savez_compressed(filename,
                                stimulus_start_times =stimulus_information[:, 0],
                                stimulus_end_times =stimulus_information[:, 1],
                                stimulus_start_indices=stimulus_information[:, 2].astype(np.uint),
                                stimulus_result_info=stimulus_information[:, 3])'''

                        # free the memory
                        self.stimulus_data[fish_index] = []
                        self.stimulus_information[fish_index] = []

                        self.shared.experiment_flow_control_currently_storing_stimulus_data[fish_index].value = 0
                        self.shared.experiment_flow_control_store_stimulus_data_completed[fish_index].value = 1


                self.stimulus_widget.taskMgr.step()  # main panda loop, needed for redrawing, etc. # vsync will take care of the sleep
                #print(time.time())
            else:
                time.sleep(0.01) #  possible sleep, can be long sometimes, very unprecise

        self.remove_module()