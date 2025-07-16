from multiprocessing import Value, sharedctypes
import ctypes
import sys

sys.path.append(r"/Users/haotianli/Code/EngertLab/Vickie_embedded_testing_nocam - Copy/modules")

from fish_fake_camera_module import FishCameraModule
from global_timer_module import GlobalTimerModule
from plotting_module import PlottingModule
from stimulus_module import StimulusModule
from experiment_flow_control_module import ExperimentFlowControlModule
import socket

class Shared():
    def __init__(self):

        computer_name = socket.gethostname()

        if computer_name == "DESKTOP-6EKL697":
            self.setup_ID = 0

        elif computer_name == "DESKTOP-J6VNU3D":
            self.setup_ID = 1

        elif computer_name == "SCRB-FC131-3":
            self.setup_ID = 2

        elif computer_name == "SCRB-FC131-4":
            self.setup_ID = 3

        else:
            self.setup_ID = 1
            
        self.fish_index_display = Value('i', 0)

        self.fish_configuration_use_fish = [Value('b', 1) for _ in range(4)]
        self.fish_configuration_ID = [Value('i', 0) for _ in range(4)]
        self.fish_configuration_genotype = [sharedctypes.RawArray(ctypes.c_ubyte, 2000) for _ in range(4)]
        self.fish_configuration_genotype_l = [Value('i', 0) for _ in range(4)]
        self.fish_configuration_age = [sharedctypes.RawArray(ctypes.c_ubyte, 2000) for _ in range(4)]
        self.fish_configuration_age_l =  [Value('i', 0) for _ in range(4)]
        self.fish_configuration_comment = [sharedctypes.RawArray(ctypes.c_ubyte, 2000) for _ in range(4)]
        self.fish_configuration_comment_l = [Value('i', 0) for _ in range(4)]

        ########################
        ### Stimulus Configuration
        self.stimulus_configuration_stimulus_path = sharedctypes.RawArray(ctypes.c_ubyte, 2000)
        self.stimulus_configuration_stimulus_path_l = Value('i', 0)
        self.stimulus_configuration_stimulus_path_update_requested = Value('b', 0)

        self.stimulus_configuration_set_x_position = [Value('d', 0) for _ in range(4)]
        self.stimulus_configuration_set_y_position = [Value('d', 0) for _ in range(4)]
        self.stimulus_configuration_set_scale = [Value('d', 0) for _ in range(4)]
        self.stimulus_configuration_set_rotation = [Value('d', 0) for _ in range(4)]

        self.stimulus_information_name = sharedctypes.RawArray(ctypes.c_ubyte, 2000)
        self.stimulus_information_name_l = Value('i', 0)
        self.stimulus_information_number_of_stimuli = Value('i', 0)
        self.stimulus_information_time_per_stimulus = Value('d', 0)

        self.stimulus_flow_control_start_requested = [Value('b', 0) for _ in range(4)]
        self.stimulus_flow_control_next_stimulus_requested = [Value('b', 0) for _ in range(4)]

        self.stimulus_flow_control_index = [Value('i', 0) for _ in range(4)]
        self.stimulus_flow_control_start_time = [Value('d', 0) for _ in range(4)]
        self.stimulus_flow_control_start_index = [Value('i', 0) for _ in range(4)]
        self.stimulus_flow_control_result_info = [Value('d', 0) for _ in range(4)]
        self.stimulus_flow_control_current_time = [Value('d', 0) for _ in range(4)]
        self.stimulus_flow_control_currently_running = [Value('b', 0) for _ in range(4)]

        ########################
        ### Camera information
        self.fish_camera_timestamp = [Value('d', 0) for _ in range(4)]
        self.fish_camera_image = [sharedctypes.RawArray(ctypes.c_ubyte, 2048 * 2048) for _ in range(4)]
        self.fish_camera_image_width = [Value('i', 0) for _ in range(4)]
        self.fish_camera_image_height = [Value('i', 0) for _ in range(4)]
        self.fish_camera_fps = [Value('d', 0) for _ in range(4)]
        self.fish_camera_shutter = [Value('d', 0) for _ in range(4)]
        self.fish_camera_gain = [Value('d', 0) for _ in range(4)]

        self.fish_camera_set_shutter = [Value('d', 2) for _ in range(4)]
        self.fish_camera_set_gain = [Value('d', 8) for _ in range(4)]
        self.fish_camera_set_roi_x = [Value('i', 0) for _ in range(4)]
        self.fish_camera_set_roi_y = [Value('i', 0) for _ in range(4)]
        self.fish_camera_set_roi_width = [Value('i', 2048) for _ in range(4)]
        self.fish_camera_set_roi_height = [Value('i', 2048) for _ in range(4)]

        self.fish_camera_update_roi_requested = [Value('b', 0) for _ in range(4)]
        self.fish_camera_update_gain_shutter_requested = [Value('b', 0) for _ in range(4)]


        ########################
        ### Eye tracking
        self.eye_tracking_configuration_left_eye_x = [Value('i', 0) for _ in range(4)]
        self.eye_tracking_configuration_left_eye_y = [Value('i', 0) for _ in range(4)]
        self.eye_tracking_configuration_right_eye_x = [Value('i', 0) for _ in range(4)]
        self.eye_tracking_configuration_right_eye_y = [Value('i', 0) for _ in range(4)]
        self.eye_tracking_configuration_threshold = [Value('i', 0) for _ in range(4)]
        self.eye_tracking_configuration_radius = [Value('i', 0) for _ in range(4)]
        self.eye_tracking_configuration_angles = [Value('i', 0) for _ in range(4)]
        self.eye_tracking_configuration_display_tracking_process = [Value('b', 1) for _ in range(4)]

        self.eye_tracking_circular_history_length_left_eye = [sharedctypes.RawArray(ctypes.c_double, 12000) for _ in range(4)]
        self.eye_tracking_circular_history_width_left_eye = [sharedctypes.RawArray(ctypes.c_double, 12000) for _ in range(4)]
        self.eye_tracking_circular_history_x_center_left_eye = [sharedctypes.RawArray(ctypes.c_double, 12000) for _ in range(4)]
        self.eye_tracking_circular_history_y_center_left_eye = [sharedctypes.RawArray(ctypes.c_double, 12000) for _ in range(4)]
        self.eye_tracking_circular_history_angle_left_eye = [sharedctypes.RawArray(ctypes.c_double, 12000) for _ in range(4)]
        
        self.eye_tracking_circular_history_length_right_eye = [sharedctypes.RawArray(ctypes.c_double, 12000) for _ in range(4)]
        self.eye_tracking_circular_history_width_right_eye = [sharedctypes.RawArray(ctypes.c_double, 12000) for _ in range(4)]
        self.eye_tracking_circular_history_x_center_right_eye = [sharedctypes.RawArray(ctypes.c_double, 12000) for _ in range(4)]
        self.eye_tracking_circular_history_y_center_right_eye = [sharedctypes.RawArray(ctypes.c_double, 12000) for _ in range(4)]
        self.eye_tracking_circular_history_angle_right_eye = [sharedctypes.RawArray(ctypes.c_double, 12000) for _ in range(4)]

        ########################
        ### Tail tracking
        #self.recorded_tail_tracking_xs = [[sharedctypes.RawArray(ctypes.c_double, 60) for _ in range(12000)] for _ in range(4)]
        self.recorded_tail_tracking_xs = [sharedctypes.RawArray(ctypes.c_double, 60*12000) for _ in range(4)]
        #print(len(self.recorded_tail_tracking_xs), len(self.recorded_tail_tracking_xs[0]),len(self.recorded_tail_tracking_xs[0][0]))
        #self.recorded_tail_tracking_ys = [[sharedctypes.RawArray(ctypes.c_double, 60) for _ in range(12000)] for _ in range(4)]
        self.recorded_tail_tracking_ys = [sharedctypes.RawArray(ctypes.c_double, 60*12000) for _ in range(4)]
        self.recorded_timeindex = [Value('i', 0) for _ in range(4)]
        self.recorded_time = [sharedctypes.RawArray(ctypes.c_double,12000) for _ in range(4)]

        self.tail_tracking_xs = [sharedctypes.RawArray(ctypes.c_double, 60) for _ in range(4)]
        self.tail_tracking_ys = [sharedctypes.RawArray(ctypes.c_double, 60) for _ in range(4)]

        self.tail_tracking_circular_history_time = [sharedctypes.RawArray(ctypes.c_double, 12000) for _ in range(4)]
        self.tail_tracking_circular_history_tail_tip_deflection = [sharedctypes.RawArray(ctypes.c_double, 12000) for _ in range(4)]
        self.tail_tracking_circular_history_tail_tip_deflection_sliding_window_variance = [sharedctypes.RawArray(ctypes.c_double, 12000) for _ in range(4)]
        self.tail_tracking_circular_history_tail_tip_deflection_sliding_window_mean = [sharedctypes.RawArray(ctypes.c_double, 12000) for _ in range(4)]
        self.tail_tracking_circular_history_bout_information = [sharedctypes.RawArray(ctypes.c_double, 12000) for _ in range(4)]
        self.tail_tracking_circular_counter = [Value('i', 0) for _ in range(4)]

        self.tail_tracking_new_beginning_bout_found = [Value('b', 0) for _ in range(4)]
        self.tail_tracking_new_completed_bout_found = [Value('b', 0) for _ in range(4)]

        self.tail_tracking_set_x0 = [Value('i', 200) for _ in range(4)]
        self.tail_tracking_set_y0 = [Value('i', 200) for _ in range(4)]
        self.tail_tracking_set_nodes = [Value('i', 38) for _ in range(4)]

        self.tail_tracking_set_bout_start_vigor = [Value('d', 1.5) for _ in range(4)]
        self.tail_tracking_set_bout_end_vigor = [Value('d', 0.5) for _ in range(4)]

        ########################
        ### Experiment Configuration
        self.experiment_configuration_storage_root_path = sharedctypes.RawArray(ctypes.c_ubyte, 2000)
        self.experiment_configuration_storage_root_path_l = Value("i", 0)
        self.experiment_configuration_number_of_trials = Value('i', 30)
        self.experiment_configuration_store_head_tail_data = Value('b', 0)
        self.experiment_configuration_store_head_tail_movie = Value('b', 0)
        self.experiment_configuration_trial_time =  Value('i', 600)

        self.experiment_flow_control_rawdata_path = [sharedctypes.RawArray(ctypes.c_ubyte, 2000) for _ in range(4)]
        self.experiment_flow_control_rawdata_path_l = [Value('i', 0) for _ in range(4)]

        self.experiment_flow_control_start_requested = Value('b', 0)
        self.experiment_flow_control_stop_requested = Value('b', 0)
        self.experiment_flow_control_currently_running = Value('b', 0)
        self.experiment_flow_control_current_trial = Value('i', 0)
        self.experiment_flow_control_percentage_done = Value('d', 0)

        self.experiment_flow_control_start_acquire_head_tail_data_requested = [Value('b', 0) for _ in range(4)]
        self.experiment_flow_control_currently_acquire_head_tail_data = [Value('b', 0) for _ in range(4)]
        self.experiment_flow_control_store_head_tail_data_requested = [Value('b', 0) for _ in range(4)]
        self.experiment_flow_control_currently_storing_head_tail_data = [Value('b', 0) for _ in range(4)]
        self.experiment_flow_control_store_head_tail_data_completed = [Value('b', 0) for _ in range(4)]

        self.experiment_flow_control_start_acquire_stimulus_data_requested = [Value('b', 0) for _ in range(4)]
        self.experiment_flow_control_currently_acquire_stimulus_data = [Value('b', 0) for _ in range(4)]
        self.experiment_flow_control_store_stimulus_data_requested = [Value('b', 0) for _ in range(4)]
        self.experiment_flow_control_currently_storing_stimulus_data = [Value('b', 0) for _ in range(4)]
        self.experiment_flow_control_store_stimulus_data_completed = [Value('b', 0) for _ in range(4)]

        ########################
        ### General programm flow
        self.global_timer = Value('d', 0)
        self.running = Value('b', 1)

    def start_threads(self):

        GlobalTimerModule(self).start()
        PlottingModule(self).start()
        StimulusModule(self).start()
        ExperimentFlowControlModule(self).start()

        for fish_index in range(4):
            FishCameraModule(self, fish_index).start()
