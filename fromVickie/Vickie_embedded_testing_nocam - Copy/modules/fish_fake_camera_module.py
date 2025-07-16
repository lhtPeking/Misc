from multiprocessing import Process
import ctypes
from numba import jit
import math
import numpy as np
import os
import cv2
import imageio
import time

class FishCameraModule(Process):
    def __init__(self, shared, fish_index):
        Process.__init__(self)

        self.fish_index = fish_index
        self.shared = shared

    
    def run(self):

        @jit(nopython=True)
        def analyze_tail(data_xs, data_ys, data_timeindex, frame, xs, ys, start_x0, start_y0, angles, kernel, vals_buffer1, vals_buffer2):

            xs = data_xs[data_timeindex,:]
            ys = data_ys[data_timeindex,:]
            '''
            dang = -90

            x_size = frame.shape[1]
            y_size = frame.shape[0]

            xs[0] = start_x0
            ys[0] = start_y0

            for i in range(1, len(xs)): # number of nodes
                for j in range(60): # 30 arc angles, spaced at 2deg

                    ang = -60 + j*2

                    x1 = int(xs[i-1] + math.cos((ang + dang) * np.pi / 180.) * 16)
                    y1 = int(ys[i-1] - math.sin((ang + dang) * np.pi / 180.) * 10)

                    if (y1 > y_size - 1 or x1 > x_size - 1 or y1 < 1 or x1 < 1):
                        vals_buffer1[j] = 0
                    else:
                        vals_buffer1[j] = frame[y1 - 1:y1 + 1, x1 - 1:x1 + 1].mean()

                    #frame[y1,x1] = 255

                # filter this with a symmetric gaussian kernel
                for j in range(60):
                    vals_buffer2[j] = 0

                    for k in range(30):
                        vals_buffer2[j] += kernel[k] * vals_buffer1[(j + k - 15) % 60]

                ang_max = -60 + np.argmin(vals_buffer2)*2 # find the blackest point

                # go to the next node
                dang += ang_max

                xs[i] = int(xs[i-1] + math.cos(dang * np.pi / 180.) * 8)
                ys[i] = int(ys[i-1] - math.sin(dang * np.pi / 180.) * 8)'''

        @jit(nopython=True)
        def analyze_bout(tail_tracking_circular_counter, circular_history_time, circular_history_tail_tip_deflection, circular_history_sliding_window_variances, circular_history_sliding_window_means, bout_finder_circular_history_bout_information, bout_vigor_thresholds_start, bout_vigor_thresholds_end):

            if circular_history_time[tail_tracking_circular_counter] > 0.05:
                # how many indezes to go to the past to reach 0.05s?
                bout_finder_past_i = 0
                while True:
                    bout_finder_past_i += 1
                    if circular_history_time[tail_tracking_circular_counter] - circular_history_time[(tail_tracking_circular_counter - bout_finder_past_i + 12000) % 12000] > 0.05:
                        break

                # get the mean within that window (in order to calculate the variance)
                windowed_mean = 0
                for i in range(bout_finder_past_i):
                    windowed_mean += circular_history_tail_tip_deflection[(tail_tracking_circular_counter - i + 12000) % 12000]
                windowed_mean /= bout_finder_past_i
                # vickie edit: save mean too
                circular_history_sliding_window_means[tail_tracking_circular_counter] = windowed_mean

                # get the variance within that window
                sliding_window_variance = 0
                for i in range(bout_finder_past_i):
                    sliding_window_variance += (circular_history_tail_tip_deflection[(tail_tracking_circular_counter - i + 12000) % 12000] - windowed_mean) ** 2
                sliding_window_variance /= bout_finder_past_i

                circular_history_sliding_window_variances[tail_tracking_circular_counter] = sliding_window_variance

                # get the bout information based on the sliding variance

                # if we were not inside a bout before or had found a bout end
                if bout_finder_circular_history_bout_information[(tail_tracking_circular_counter - 1 + 12000) % 12000] == 0 or bout_finder_circular_history_bout_information[(tail_tracking_circular_counter - 1 + 12000) % 12000] == 3:

                    # and if our tail vigor is now above a threshold ...
                    if sliding_window_variance >= bout_vigor_thresholds_start:
                        bout_finder_circular_history_bout_information[tail_tracking_circular_counter] = 1 # then now we have a new bout
                    else:
                        bout_finder_circular_history_bout_information[tail_tracking_circular_counter] = 0 # then we are now or stay outside a bout

                # if we just had found a bout start or are inside a bout
                if bout_finder_circular_history_bout_information[(tail_tracking_circular_counter - 1 + 12000) % 12000] == 1 or bout_finder_circular_history_bout_information[(tail_tracking_circular_counter - 1 + 12000) % 12000] == 2:

                    # and if the tail vigor drops below the lower threshold
                    if sliding_window_variance <= bout_vigor_thresholds_end:

                        bout_finder_circular_history_bout_information[tail_tracking_circular_counter] = 3 # then we found a bout end
                    else:
                        bout_finder_circular_history_bout_information[tail_tracking_circular_counter] = 2 # alternatively, we remain inside the bout

        self.tail_tracking_circular_counter = 0
        self.fish_camera_image = np.ctypeslib.as_array(self.shared.fish_camera_image[self.fish_index])    

        self.tail_tracking_circular_history_time = np.ctypeslib.as_array(self.shared.tail_tracking_circular_history_time[self.fish_index])
        self.tail_tracking_circular_history_tail_tip_deflection = np.ctypeslib.as_array(self.shared.tail_tracking_circular_history_tail_tip_deflection[self.fish_index])
        self.tail_tracking_circular_history_tail_tip_deflection_sliding_window_variance = np.ctypeslib.as_array(self.shared.tail_tracking_circular_history_tail_tip_deflection_sliding_window_variance[self.fish_index])
        self.tail_tracking_circular_history_tail_tip_deflection_sliding_window_mean = np.ctypeslib.as_array(self.shared.tail_tracking_circular_history_tail_tip_deflection_sliding_window_mean[self.fish_index])
        self.tail_tracking_circular_history_bout_information = np.ctypeslib.as_array(self.shared.tail_tracking_circular_history_bout_information[self.fish_index])

        self.recorded_tail_tracking_xs = np.ctypeslib.as_array(self.shared.recorded_tail_tracking_xs[self.fish_index])
        self.recorded_tail_tracking_ys = np.ctypeslib.as_array(self.shared.recorded_tail_tracking_ys[self.fish_index])
        self.tail_tracking_xs = np.ctypeslib.as_array(self.shared.tail_tracking_xs[self.fish_index])
        self.tail_tracking_ys = np.ctypeslib.as_array(self.shared.tail_tracking_ys[self.fish_index])
        self.recorded_time = np.ctypeslib.as_array(self.shared.recorded_time[self.fish_index])

        
        x = np.arange(-15, 15)
        kernel = np.exp(-(x ** 2) / (20 ** 2))
        kernel /= np.sum(kernel)

        vals_buffer1 = np.zeros(60)
        vals_buffer2 = np.zeros(60)


        while self.shared.running.value == 1:

            #try:
            #    self.shared.fish_camera_timestamp[self.fish_index].value = self.camera_library.get_image(self.fish_roi_buffer, self.set_roi_width*self.set_roi_height)

            #except Exception as e:
            #    print("Tail camera error:", e)
            #    continue
            #time.sleep(0.0025) # in order to run at 400 hz, we must run at full power, no sleep possible

            self.set_roi_width = int(self.shared.fish_camera_set_roi_width[self.fish_index].value / 32) * 32
            self.set_roi_height = int(self.shared.fish_camera_set_roi_height[self.fish_index].value / 32) * 32
            self.fish_roi_buffer = ctypes.create_string_buffer(self.set_roi_width * self.set_roi_height)
            frame = np.fromstring(self.fish_roi_buffer, dtype=np.uint8).reshape((self.set_roi_height, self.set_roi_width))

            ####################
            # Tail tracking
            nodes = self.shared.tail_tracking_set_nodes[self.fish_index].value
            currtime = self.shared.recorded_timeindex[self.fish_index].value
            if currtime%1000 == 0: print(currtime)
            #xs = np.zeros(tail_tracking_set_nodes)
            #ys = np.zeros(tail_tracking_set_nodes)
            xs = self.recorded_tail_tracking_xs[nodes*currtime:nodes*(currtime+1)].copy()
            ys = self.recorded_tail_tracking_ys[nodes*currtime:nodes*(currtime+1)].copy()
            #if currtime % 1000 == 0:
            #    print(currtime, xs)
            #if (self.fish_index == 0) & (currtime < 10):
            #    print(currtime,xs,ys)
            #print('hoooo', xs)
            #print(self.shared.recorded_timeindex.value)

            #print('hallo', frame.shape[0], frame.shape[1])
            # get the tail shape
            #analyze_tail(self.shared.recorded_tail_tracking_xs[self.fish_index], self.shared.recorded_tail_tracking_ys[self.fish_index], self.shared.recorded_timeindex, frame, xs, ys, self.shared.tail_tracking_set_x0[self.fish_index].value, self.shared.tail_tracking_set_y0[self.fish_index].value, 120, kernel, vals_buffer1, vals_buffer2)

            # angle to head ar the average of the last 5 nodes averages
            angles_to_head = 180 / np.pi * np.arctan2(xs - xs[0], ys - ys[0])

            tail_tip_deflection = np.mean(angles_to_head[-5:]) # or average the last xs, ys?

            self.tail_tracking_circular_history_time[self.tail_tracking_circular_counter] = self.shared.global_timer.value
            self.tail_tracking_circular_history_tail_tip_deflection[self.tail_tracking_circular_counter] = tail_tip_deflection

            # analyze the bouts (look for vigor thresholds)
            '''
            analyze_bout(self.tail_tracking_circular_counter,
                         self.tail_tracking_circular_history_time,
                         self.tail_tracking_circular_history_tail_tip_deflection,
                         self.tail_tracking_circular_history_tail_tip_deflection_sliding_window_variance,
                         self.tail_tracking_circular_history_tail_tip_deflection_sliding_window_mean,
                         self.tail_tracking_circular_history_bout_information,
                         self.shared.tail_tracking_set_bout_start_vigor[self.fish_index].value, self.shared.tail_tracking_set_bout_end_vigor[self.fish_index].value)

            if self.tail_tracking_circular_history_bout_information[self.tail_tracking_circular_counter] == 1:
                self.shared.tail_tracking_new_beginning_bout_found[self.fish_index].value = 1  # this needs to be reset by the process that cares about this, likely the visual stimulus

            if self.tail_tracking_circular_history_bout_information[self.tail_tracking_circular_counter] == 3:
                self.shared.tail_tracking_new_completed_bout_found[self.fish_index].value = 1  # this needs to be reset by the process that cares about this, likely the visual stimulus'''

            ######
            # Dataset for this frame is complete, share the all the information information with the global world
            self.fish_camera_image[:frame.shape[0] * frame.shape[1]] = frame.flatten()
            self.shared.fish_camera_image_width[self.fish_index].value = frame.shape[0]
            self.shared.fish_camera_image_height[self.fish_index].value = frame.shape[1]
            self.tail_tracking_xs[:len(xs)] = xs
            self.tail_tracking_ys[:len(ys)] = ys

            # advancing the local counter must still happen after data storage
            self.shared.tail_tracking_circular_counter[self.fish_index].value = self.tail_tracking_circular_counter  # copy the circular counter to the displaying task before
            #while(self.shared.global_timer.value < self.recorded_time[currtime]): 
            #if self.tail_tracking_circular_history_time[self.tail_tracking_circular_counter] - self.tail_tracking_circular_history_time[self.tail_tracking_circular_counter-1] > 1/400:
            #if self.fish_index == 0: print(np.maximum(0, 1/400 - (self.tail_tracking_circular_history_time[self.tail_tracking_circular_counter] - self.tail_tracking_circular_history_time[self.tail_tracking_circular_counter-1]))/1000)
            time.sleep(np.maximum(0, 1/400 - (self.tail_tracking_circular_history_time[self.tail_tracking_circular_counter] - self.tail_tracking_circular_history_time[self.tail_tracking_circular_counter-1]))/1000)
            self.shared.recorded_timeindex[self.fish_index].value += 1

            '''
            #########################
            # Data storage
            if self.shared.experiment_flow_control_start_acquire_head_tail_data_requested[self.fish_index].value == 1:
                self.shared.experiment_flow_control_start_acquire_head_tail_data_requested[self.fish_index].value = 0

                self.head_tail_movie = []
                self.head_tail_data = []
                self.tail_shape_data = []

                self.shared.experiment_flow_control_currently_acquire_head_tail_data[self.fish_index].value = 1 # start adding data to these empty lists


            if self.shared.experiment_flow_control_currently_acquire_head_tail_data[self.fish_index].value == 1:

                self.head_tail_data.append([self.shared.global_timer.value,
                                            self.tail_tracking_circular_history_tail_tip_deflection[self.tail_tracking_circular_counter],
                                            self.tail_tracking_circular_history_tail_tip_deflection_sliding_window_variance[self.tail_tracking_circular_counter],
                                            self.tail_tracking_circular_history_tail_tip_deflection_sliding_window_mean[self.tail_tracking_circular_counter],
                                            self.tail_tracking_circular_history_bout_information[self.tail_tracking_circular_counter]]) # 0: outside bout, 1: bout start, 2: inside bout, 3: bout stop

                self.tail_shape_data.append([xs, ys])


            if self.shared.experiment_flow_control_store_head_tail_data_requested[self.fish_index].value == 1:
                self.shared.experiment_flow_control_store_head_tail_data_requested[self.fish_index].value = 0

                self.shared.experiment_flow_control_currently_storing_head_tail_data[self.fish_index].value = 1
                self.shared.experiment_flow_control_currently_acquire_head_tail_data[self.fish_index].value = 0

                # save the tail data from that trial
                rawdata_path = bytearray(self.shared.experiment_flow_control_rawdata_path[self.fish_index][:self.shared.experiment_flow_control_rawdata_path_l[self.fish_index].value]).decode()

                head_tail_data = np.array(self.head_tail_data)
                tail_shape_data = np.array(self.tail_shape_data)

                if self.shared.experiment_configuration_store_head_tail_data.value == 1:

                    filename = os.path.join(rawdata_path, "trial{:03d}_head_tail_data".format(self.shared.experiment_flow_control_current_trial.value))
                    """
                    np.savez_compressed(filename,
                                        camera_time=head_tail_data[:, 0],
                                        left_eye_angle=head_tail_data[:, 1],
                                        right_eye_angle=head_tail_data[:, 2],
                                        tail_tip_deflection=head_tail_data[:, 3],
                                        tail_vigor=head_tail_data[:, 4],
                                        tail_bout_information=head_tail_data[:, 5].astype(np.uint),
                                        tail_shape_xs=tail_shape_data[:, 0, :].astype(np.uint16),
                                        tail_shape_ys=tail_shape_data[:, 1, :].astype(np.uint16))"""


                if self.shared.experiment_configuration_store_head_tail_movie.value == 1:

                    filename = os.path.join(rawdata_path, "trial{:03d}_head_tail_movie.mp4".format(self.shared.experiment_flow_control_current_trial.value))

                    """writer = imageio.get_writer(filename, fps=60)

                    for i in range(len(self.head_tail_data)):
                        # draw the points
                        frame = self.head_tail_movie[i]

                        for j in range(len(self.tail_shape_data[i][0])):

                            cv2.circle(frame, (int(self.tail_shape_data[i][0][j]), int(self.tail_shape_data[i][1][j])), 2, thickness=-1, color=255)
                            cv2.putText(frame, "Time: {:.3f} s".format(self.head_tail_data[i][0]), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=255)

                        writer.append_data(frame)

                    writer.close()"""

                # free the memory
                self.head_tail_movie = []
                self.head_tail_data = []
                self.tail_shape_data = []

                self.shared.experiment_flow_control_currently_storing_head_tail_data[self.fish_index].value = 0
                self.shared.experiment_flow_control_store_head_tail_data_completed[self.fish_index].value = 1

            '''

            self.tail_tracking_circular_counter += 1
            #print(self.fish_index, self.tail_tracking_circular_counter)
            if self.tail_tracking_circular_counter == 12000:
                self.tail_tracking_circular_counter = 0

