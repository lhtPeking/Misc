from multiprocessing import Process
import time
import os
import shutil
import pickle
import numpy as np

class ExperimentFlowControlModule(Process):
    def __init__(self, shared):
        Process.__init__(self)

        self.shared = shared

    def run(self):

        while self.shared.running.value == 1:

            if self.shared.experiment_flow_control_start_requested.value == 1:
                self.shared.experiment_flow_control_start_requested.value = 0

                self.shared.experiment_flow_control_currently_running.value = 1

                # Make the folder structure for all fish
                root_path = bytearray(self.shared.experiment_configuration_storage_root_path[:self.shared.experiment_configuration_storage_root_path_l.value]).decode()
                '''
                try:
                    os.mkdir(root_path)
                except: # main folder likely already exists
                    pass
                '''
                stimulus_name = bytearray(self.shared.stimulus_information_name[:self.shared.stimulus_information_name_l.value]).decode()
                root_path = os.path.join(root_path, stimulus_name)

                '''
                try:
                    os.mkdir(root_path)
                except: # folder for the stimulus might also already exist
                    pass'''

                for fish_index in range(4):

                    if self.shared.fish_configuration_use_fish[fish_index].value == 0:
                        continue

                    path = os.path.join(root_path, time.strftime("%Y-%m-%d_%H-%M-%S") + "_fish{:03d}".format(self.shared.fish_configuration_ID[fish_index].value))
                    '''
                    try:
                        os.mkdir(path)
                    except:
                        print("Experiment folder", path, "already exists. Trials will get overwritten.")
                        continue
                    '''
                    rawdata_path = os.path.join(path, "rawdata")
                    '''
                    os.mkdir(rawdata_path)
                    '''
                    self.shared.experiment_flow_control_rawdata_path[fish_index][:len(rawdata_path)] = rawdata_path.encode()
                    self.shared.experiment_flow_control_rawdata_path_l[fish_index].value = len(rawdata_path)

                    ########################
                    # Save general experiment information
                    position = ["Upper left", "Upper right", "Lower left", "Lower right"][fish_index]

                    experiment_information = "Fish Index: {} ({})\n".format(fish_index, position)
                    experiment_information += "Date and time: {}\n\n".format(time.strftime("%Y-%m-%d_%H-%M-%S"))
                    experiment_information += "fish_configuration_ID: {}\n".format(self.shared.fish_configuration_ID[fish_index].value)
                    experiment_information += "fish_configuration_genotype: {}\n".format(bytearray(self.shared.fish_configuration_genotype[fish_index][:self.shared.fish_configuration_genotype_l[fish_index].value]).decode())
                    experiment_information += "fish_configuration_age: {}\n".format(bytearray(self.shared.fish_configuration_age[fish_index][:self.shared.fish_configuration_age_l[fish_index].value]).decode())
                    experiment_information += "fish_configuration_comment: {}\n".format(bytearray(self.shared.fish_configuration_comment[fish_index][:self.shared.fish_configuration_comment_l[fish_index].value]).decode())

                    #fp = open(os.path.join(path, "experiment_information.txt"), "w")
                    #fp.write(experiment_information)
                    #fp.close()

                    ################
                    # Save the stimulus python file also
                    full_path_to_module = bytearray(self.shared.stimulus_configuration_stimulus_path[:self.shared.stimulus_configuration_stimulus_path_l.value]).decode()
                    #shutil.copyfile(full_path_to_module, os.path.join(path, "stimulus.py"))

                self.go_stop = False

                for trial in range(self.shared.experiment_configuration_number_of_trials.value):

                    self.shared.experiment_flow_control_current_trial.value = trial

                    # Tell the other modules to start storing data
                    for fish_index in range(4):

                        if self.shared.fish_configuration_use_fish[fish_index].value == 0:
                            continue

                        self.shared.experiment_flow_control_start_acquire_head_tail_data_requested[fish_index].value = 1
                        self.shared.experiment_flow_control_start_acquire_stimulus_data_requested[fish_index].value = 1

                    for fish_index in range(4): # start all the fish

                        if self.shared.fish_configuration_use_fish[fish_index].value == 0:
                            continue
                        
                        # print("self.shared.stimulus_information_number_of_stimuli.value:", self.shared.stimulus_information_number_of_stimuli.value)
                        # self.shared.stimulus_information_number_of_stimuli.value += 1
                        self.shared.stimulus_flow_control_index[fish_index].value = np.random.randint(self.shared.stimulus_information_number_of_stimuli.value) # start with a random stimulus
                        self.shared.stimulus_flow_control_start_requested[fish_index].value = 1

                    # Waiting loop to the end of the trial
                    trial_start_time = self.shared.global_timer.value

                    while self.shared.global_timer.value - trial_start_time < self.shared.experiment_configuration_trial_time.value:
                        time.sleep(0.05)

                        # Calculate the percentage done of the experiment
                        total_time = self.shared.experiment_configuration_number_of_trials.value * self.shared.experiment_configuration_trial_time.value

                        current_time = trial * self.shared.experiment_configuration_trial_time.value + self.shared.global_timer.value - trial_start_time

                        self.shared.experiment_flow_control_percentage_done.value = 100 * current_time / total_time

                        if self.shared.experiment_flow_control_stop_requested.value == 1:
                            self.shared.experiment_flow_control_stop_requested.value = 0

                            self.go_stop = True
                            break

                    for fish_index in range(4):

                        if self.shared.fish_configuration_use_fish[fish_index].value == 0:
                            continue

                        self.shared.stimulus_flow_control_currently_running[fish_index].value = 0 # Stop the stimulus displaying and updating at the end of the trial

                    if self.go_stop:
                        break

                    # Stop all acquisitions
                    for fish_index in range(4):

                        if self.shared.fish_configuration_use_fish[fish_index].value == 0:
                            continue

                        # stop the tail tracking and visual stimulus data storage
                        self.shared.experiment_flow_control_currently_acquire_head_tail_data[fish_index].value = 0
                        self.shared.experiment_flow_control_currently_acquire_stimulus_data[fish_index].value = 0

                    if self.go_stop:
                        break

                    # Tell all modules to store data
                    for fish_index in range(4):

                        if self.shared.fish_configuration_use_fish[fish_index].value == 0:
                            continue

                        # tell the modules also to save the data now. Later we wait until it is finished
                        self.shared.experiment_flow_control_store_head_tail_data_completed[fish_index].value = 0
                        self.shared.experiment_flow_control_store_stimulus_data_completed[fish_index].value = 0

                        self.shared.experiment_flow_control_store_head_tail_data_requested[fish_index].value = 1
                        self.shared.experiment_flow_control_store_stimulus_data_requested[fish_index].value = 1

                    # Wait until all fish finished saving
                    for fish_index in range(4):

                        if self.shared.fish_configuration_use_fish[fish_index].value == 0:
                            continue

                        while self.shared.experiment_flow_control_store_head_tail_data_completed[fish_index].value == 0:
                            pass

                        while self.shared.experiment_flow_control_store_stimulus_data_completed[fish_index].value == 0:
                            pass

                    if self.go_stop:
                        break

                self.shared.experiment_flow_control_currently_running.value = 0