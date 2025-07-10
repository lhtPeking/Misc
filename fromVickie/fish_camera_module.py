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

    def open_cam(self):

        self.set_roi_width = int(self.shared.fish_camera_set_roi_width[self.fish_index].value / 32) * 32
        self.set_roi_height = int(self.shared.fish_camera_set_roi_height[self.fish_index].value / 32) * 32

        if self.set_roi_width < 32:
            self.set_roi_width = 32

        if self.set_roi_height < 32:
            self.set_roi_height = 32

        if self.set_roi_width > 2048:
            self.set_roi_width = 2048

        if self.set_roi_height > 2048:
            self.set_roi_height = 2048

        self.set_roi_x = self.shared.fish_camera_set_roi_x[self.fish_index].value
        self.set_roi_y = self.shared.fish_camera_set_roi_y[self.fish_index].value

        if self.set_roi_x < 0:
            self.set_roi_x = 0

        if self.set_roi_y < 0:
            self.set_roi_y = 0

        if self.set_roi_x > 2048 - self.set_roi_width:
            self.set_roi_x = 2048 - self.set_roi_width

        if self.set_roi_y > 2048 - self.set_roi_height:
            self.set_roi_y = 2048 - self.set_roi_height

        self.set_roi_x = int(self.set_roi_x / 32) * 32
        self.set_roi_y = int(self.set_roi_y / 32) * 32

        self.fish_roi_buffer = ctypes.create_string_buffer(self.set_roi_width * self.set_roi_height)
        self.tail_tracking_circular_counter = 0
        self.fps_timer_start = 0

        if self.shared.setup_ID == 0:
            if self.fish_index == 0:
                self.camera_serial = 19044564
            elif self.fish_index == 1:
                self.camera_serial = 19044553
            elif self.fish_index == 2:
                self.camera_serial = 19044554
            elif self.fish_index == 3:
                self.camera_serial = 19044567

        if self.shared.setup_ID == 1:
            if self.fish_index == 0:
                self.camera_serial = 18570322
            elif self.fish_index == 1:
                self.camera_serial = 18570319
            elif self.fish_index == 2:
                self.camera_serial = 18570321
            elif self.fish_index == 3:
                self.camera_serial = 18570323

        if self.shared.setup_ID == 2:
            if self.fish_index == 0:
                self.camera_serial = 19337760
            elif self.fish_index == 1:
                self.camera_serial = 19337711
            elif self.fish_index == 2:
                self.camera_serial = 19337715
            elif self.fish_index == 3:
                self.camera_serial = 19337714

        if self.shared.setup_ID == 3:
            if self.fish_index == 0:
                self.camera_serial = 19337705
            elif self.fish_index == 1:
                self.camera_serial = 19337713
            elif self.fish_index == 2:
                self.camera_serial = 19337708
            elif self.fish_index == 3:
                self.camera_serial = 19337712

        #if self.shared.setup_ID == -1:
        #    self.camera_opened = False
        #else:
        self.camera_opened = self.camera_library.open_cam(ctypes.c_uint32(self.camera_serial), ctypes.c_uint32(0),
                                     ctypes.c_uint32(self.set_roi_x), ctypes.c_uint32(self.set_roi_y),
                                     ctypes.c_uint32(self.set_roi_width), ctypes.c_uint32(self.set_roi_height),
                                     ctypes.c_double(self.shared.fish_camera_set_gain[self.fish_index].value),
                                     ctypes.c_double(self.shared.fish_camera_set_shutter[self.fish_index].value))

    def run(self):

        @jit
        def get_eye_shape(x0, y0, frame, threshold, arc_radius, arc_angles, points_found, xs_buffer, ys_buffer, draw):

            points_found[0] = 0  # we did not find the contour

            x_size = frame.shape[0]
            y_size = frame.shape[1]

            # first, walk up and find the first intersection
            for i in range(300):
                if y0 + i >= y_size:  # might hit the image corner
                    return

                if frame[x0, y0 + i] > threshold:
                    break

                if draw == 1:
                    frame[x0, y0 + i] = 0  # for debugging

            if i == 299:  # nothing found
                return

            start_x0 = x0
            start_y0 = y0 + i

            ang0 = 0
            x0 = start_x0
            y0 = start_y0

            xs_buffer[0] = x0
            ys_buffer[0] = y0

            for j in range(1, 500):  # max, likely less

                dang = -np.pi / 2
                while dang < -np.pi / 2 + arc_angles * np.pi / 180:

                    # walk a bigger radius
                    x0_ = int(x0 + arc_radius * math.cos(ang0 + dang))
                    y0_ = int(y0 + arc_radius * math.sin(ang0 + dang))

                    if x0_ >= x_size or y0_ >= y_size or x0_ < 0 or y0_ < 0:
                        return  # we should never read the border, ... bad

                    if frame[x0_, y0_] > threshold:
                        break

                    if draw == 1:
                        frame[x0_, y0_] = 0  # for debugging

                    dang += np.pi / 180

                # make a smaller step in that direction
                x0 = int(x0 + 3 * math.cos(ang0 + dang))
                y0 = int(y0 + 3 * math.sin(ang0 + dang))

                ang0 += dang

                if draw == 1:
                    frame[x0, y0] = 0  # for debugging

                xs_buffer[j] = x0
                ys_buffer[j] = y0

                # check if we have already been here
                if j > 10:
                    if math.sqrt((x0 - start_x0) ** 2 + (y0 - start_y0) ** 2) < arc_radius:  # no more step needed
                        break

            points_found[0] = j

        @jit
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

        @jit
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


        self.fish_camera_image = np.ctypeslib.as_array(self.shared.fish_camera_image[self.fish_index])

        self.eye_tracking_circular_history_x_center_left_eye = np.ctypeslib.as_array(self.shared.eye_tracking_circular_history_x_center_left_eye[self.fish_index])
        self.eye_tracking_circular_history_y_center_left_eye = np.ctypeslib.as_array(self.shared.eye_tracking_circular_history_y_center_left_eye[self.fish_index])
        self.eye_tracking_circular_history_length_left_eye = np.ctypeslib.as_array(self.shared.eye_tracking_circular_history_length_left_eye[self.fish_index])
        self.eye_tracking_circular_history_width_left_eye = np.ctypeslib.as_array(self.shared.eye_tracking_circular_history_width_left_eye[self.fish_index])
        self.eye_tracking_circular_history_angle_left_eye = np.ctypeslib.as_array(self.shared.eye_tracking_circular_history_angle_left_eye[self.fish_index])
        
        self.eye_tracking_circular_history_x_center_right_eye = np.ctypeslib.as_array(self.shared.eye_tracking_circular_history_x_center_right_eye[self.fish_index])
        self.eye_tracking_circular_history_y_center_right_eye = np.ctypeslib.as_array(self.shared.eye_tracking_circular_history_y_center_right_eye[self.fish_index])
        self.eye_tracking_circular_history_length_right_eye = np.ctypeslib.as_array(self.shared.eye_tracking_circular_history_length_right_eye[self.fish_index])
        self.eye_tracking_circular_history_width_right_eye = np.ctypeslib.as_array(self.shared.eye_tracking_circular_history_width_right_eye[self.fish_index])
        self.eye_tracking_circular_history_angle_right_eye = np.ctypeslib.as_array(self.shared.eye_tracking_circular_history_angle_right_eye[self.fish_index])
    

        self.tail_tracking_circular_history_time = np.ctypeslib.as_array(self.shared.tail_tracking_circular_history_time[self.fish_index])
        self.tail_tracking_circular_history_tail_tip_deflection = np.ctypeslib.as_array(self.shared.tail_tracking_circular_history_tail_tip_deflection[self.fish_index])
        self.tail_tracking_circular_history_tail_tip_deflection_sliding_window_variance = np.ctypeslib.as_array(self.shared.tail_tracking_circular_history_tail_tip_deflection_sliding_window_variance[self.fish_index])
        self.tail_tracking_circular_history_tail_tip_deflection_sliding_window_mean = np.ctypeslib.as_array(self.shared.tail_tracking_circular_history_tail_tip_deflection_sliding_window_mean[self.fish_index])
        self.tail_tracking_circular_history_bout_information = np.ctypeslib.as_array(self.shared.tail_tracking_circular_history_bout_information[self.fish_index])

        self.tail_tracking_xs = np.ctypeslib.as_array(self.shared.tail_tracking_xs[self.fish_index])
        self.tail_tracking_ys = np.ctypeslib.as_array(self.shared.tail_tracking_ys[self.fish_index])

        # buffers for the eye tracking
        points_found = [0]
        xs_buffer_left_eye = np.zeros(500)
        ys_buffer_left_eye = np.zeros(500)
        xs_buffer_right_eye = np.zeros(500)
        ys_buffer_right_eye = np.zeros(500)

        # open the point gray camera library
        python_file_path = os.path.dirname(os.path.abspath(__file__))

        self.camera_library = ctypes.cdll.LoadLibrary(r"C:\Users\vicki\OneDrive - Harvard University\Engert Lab\Explore_Exploit\Code_backupscripts\Vickie_embedded_testing_nocam\my_helpers_old\flir_camera_helper_c\x64\Release\flir_camera_helper_c.dll")

        self.camera_library.open_cam.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_double,
            ctypes.c_double
        ]

        self.camera_library.get_gain.restype = ctypes.c_double
        self.camera_library.get_shutter.restype = ctypes.c_double
        self.camera_library.get_image.restype = ctypes.c_double


        x = np.arange(-15, 15)
        kernel = np.exp(-(x ** 2) / (20 ** 2))
        kernel /= np.sum(kernel)

        vals_buffer1 = np.zeros(60)
        vals_buffer2 = np.zeros(60)

        self.camera_library.init()

        self.open_cam()

        if self.camera_opened == False:
            print("Cannot open camera", self.fish_index)
            return

        while self.shared.running.value == 1:

            self.shared.fish_camera_gain[self.fish_index].value = self.camera_library.get_gain()
            self.shared.fish_camera_shutter[self.fish_index].value = self.camera_library.get_shutter()

            if self.shared.fish_camera_update_gain_shutter_requested[self.fish_index].value == 1:
                self.shared.fish_camera_update_gain_shutter_requested[self.fish_index].value = 0

                self.camera_library.set_gain(ctypes.c_double(self.shared.fish_camera_set_gain[self.fish_index].value))
                self.camera_library.set_shutter(ctypes.c_double(self.shared.fish_camera_set_shutter[self.fish_index].value))

            if self.shared.fish_camera_update_roi_requested[self.fish_index].value == 1:
                self.shared.fish_camera_update_roi_requested[self.fish_index].value = 0

                self.camera_library.close_cam()
                self.open_cam()

                if self.camera_opened == False:
                    print("Cannot open camera", self.fish_index)
                    return

            #time.sleep(0.001) # in order to run at 400 hz, we must run at full power, no sleep possible
            try:
                self.shared.fish_camera_timestamp[self.fish_index].value = self.camera_library.get_image(self.fish_roi_buffer, self.set_roi_width*self.set_roi_height)
                frame = np.fromstring(self.fish_roi_buffer, dtype=np.uint8).reshape((self.set_roi_height, self.set_roi_width))

            except Exception as e:
                print("Tail camera error:", e)
                continue

            ########################
            # Eye tracking
            # Get the eye angle of the left eye
            flipped_frame = (np.flipud(frame).T).copy()

            get_eye_shape(self.shared.eye_tracking_configuration_left_eye_x[self.fish_index].value, self.set_roi_height - self.shared.eye_tracking_configuration_left_eye_y[self.fish_index].value, flipped_frame,
                          self.shared.eye_tracking_configuration_threshold[self.fish_index].value, self.shared.eye_tracking_configuration_radius[self.fish_index].value, self.shared.eye_tracking_configuration_angles[self.fish_index].value,
                          points_found, xs_buffer_left_eye, ys_buffer_left_eye,  self.shared.eye_tracking_configuration_display_tracking_process[self.fish_index].value)

            if points_found[0] >= 5:  # did not find the contour
                xs_left_eye = xs_buffer_left_eye[:points_found[0]]
                ys_left_eye = ys_buffer_left_eye[:points_found[0]]

                ellipse = cv2.fitEllipse(np.c_[xs_left_eye, ys_left_eye].astype(np.float32))

                x_center_left_eye = ellipse[0][0]
                y_center_left_eye = ellipse[0][1]
                length_left_eye = ellipse[1][1]  # length of ellipse
                width_left_eye = ellipse[1][0]  # width of ellipse
                orientation_left_eye = ellipse[2]

                self.eye_tracking_circular_history_x_center_left_eye[self.tail_tracking_circular_counter] = x_center_left_eye
                self.eye_tracking_circular_history_y_center_left_eye[self.tail_tracking_circular_counter] = y_center_left_eye
                self.eye_tracking_circular_history_length_left_eye[self.tail_tracking_circular_counter] = length_left_eye
                self.eye_tracking_circular_history_width_left_eye[self.tail_tracking_circular_counter] = width_left_eye

                if 90 - orientation_left_eye < 0:
                    self.eye_tracking_circular_history_angle_left_eye[self.tail_tracking_circular_counter] = 270 - orientation_left_eye
                else:
                    self.eye_tracking_circular_history_angle_left_eye[self.tail_tracking_circular_counter] = 90 - orientation_left_eye

                if self.shared.eye_tracking_configuration_display_tracking_process[self.fish_index].value == 1:
                    a = np.array([np.c_[ys_left_eye, xs_left_eye]]).astype(np.int).swapaxes(0, 1)
                    cv2.drawContours(flipped_frame, [a], -1, 200, 1)

                    dx = 0.5 * length_left_eye * np.cos((90 + orientation_left_eye) * np.pi / 180)
                    dy = 0.5 * length_left_eye * np.sin((90 + orientation_left_eye) * np.pi / 180)
                    cv2.line(flipped_frame, (int(y_center_left_eye - dy), int(x_center_left_eye - dx)), (int(y_center_left_eye + dy), int(x_center_left_eye + dx)), 255, 1)

                    dx = 0.5 * width_left_eye * np.cos(orientation_left_eye * np.pi / 180)
                    dy = 0.5 * width_left_eye * np.sin(orientation_left_eye * np.pi / 180)
                    cv2.line(flipped_frame, (int(y_center_left_eye - dy), int(x_center_left_eye - dx)), (int(y_center_left_eye + dy), int(x_center_left_eye + dx)), 0, 1)

            # Get the eye angle of the right eye
            get_eye_shape(self.shared.eye_tracking_configuration_right_eye_x[self.fish_index].value, self.set_roi_height - self.shared.eye_tracking_configuration_right_eye_y[self.fish_index].value, flipped_frame,
                          self.shared.eye_tracking_configuration_threshold[self.fish_index].value, self.shared.eye_tracking_configuration_radius[self.fish_index].value, self.shared.eye_tracking_configuration_angles[self.fish_index].value,
                          points_found, xs_buffer_right_eye, ys_buffer_right_eye,  self.shared.eye_tracking_configuration_display_tracking_process[self.fish_index].value)

            if points_found[0] >= 5:  # did not find the contour
                xs_right_eye = xs_buffer_right_eye[:points_found[0]]
                ys_right_eye = ys_buffer_right_eye[:points_found[0]]

                ellipse = cv2.fitEllipse(np.c_[xs_right_eye, ys_right_eye].astype(np.float32))

                x_center_right_eye = ellipse[0][0]
                y_center_right_eye = ellipse[0][1]
                length_right_eye = ellipse[1][1]  # length of ellipse
                width_right_eye = ellipse[1][0]  # width of ellipse
                orientation_right_eye = ellipse[2]

                self.eye_tracking_circular_history_x_center_right_eye[self.tail_tracking_circular_counter] = x_center_right_eye
                self.eye_tracking_circular_history_y_center_right_eye[self.tail_tracking_circular_counter] = y_center_right_eye
                self.eye_tracking_circular_history_length_right_eye[self.tail_tracking_circular_counter] = length_right_eye
                self.eye_tracking_circular_history_width_right_eye[self.tail_tracking_circular_counter] = width_right_eye

                if 90 - orientation_right_eye < 0:
                    self.eye_tracking_circular_history_angle_right_eye[self.tail_tracking_circular_counter] = 270 - orientation_right_eye
                else:
                    self.eye_tracking_circular_history_angle_right_eye[self.tail_tracking_circular_counter] = 90 - orientation_right_eye

                if self.shared.eye_tracking_configuration_display_tracking_process[self.fish_index].value == 1:
                    a = np.array([np.c_[ys_right_eye, xs_right_eye]]).astype(np.int).swapaxes(0, 1)
                    cv2.drawContours(flipped_frame, [a], -1, 200, 1)

                    dx = 0.5 * length_right_eye * np.cos((90 + orientation_right_eye) * np.pi / 180)
                    dy = 0.5 * length_right_eye * np.sin((90 + orientation_right_eye) * np.pi / 180)
                    cv2.line(flipped_frame, (int(y_center_right_eye - dy), int(x_center_right_eye - dx)), (int(y_center_right_eye + dy), int(x_center_right_eye + dx)), 255, 1)

                    dx = 0.5 * width_right_eye * np.cos(orientation_right_eye * np.pi / 180)
                    dy = 0.5 * width_right_eye * np.sin(orientation_right_eye * np.pi / 180)
                    cv2.line(flipped_frame, (int(y_center_right_eye - dy), int(x_center_right_eye - dx)), (int(y_center_right_eye + dy), int(x_center_right_eye + dx)), 0, 1)

            frame = np.flipud((flipped_frame.T)).copy()

            ####################
            # Tail tracking
            tail_tracking_set_nodes = self.shared.tail_tracking_set_nodes[self.fish_index].value

            #xs = np.zeros(tail_tracking_set_nodes)
            #ys = np.zeros(tail_tracking_set_nodes)

            xs = self.shared.recorded_tail_tracking_xs[self.fish_index][:,self.shared.recorded_timeindex]
            ys = self.shared.recorded_tail_tracking_ys[self.fish_index][:,self.shared.recorded_timeindex]

            print('hallo', frame.shape[0], frame.shape[1])
            # get the tail shape
            #analyze_tail(self.shared.recorded_tail_tracking_xs[self.fish_index], self.shared.recorded_tail_tracking_ys[self.fish_index], self.shared.recorded_timeindex, frame, xs, ys, self.shared.tail_tracking_set_x0[self.fish_index].value, self.shared.tail_tracking_set_y0[self.fish_index].value, 120, kernel, vals_buffer1, vals_buffer2)

            # angle to head ar the average of the last 5 nodes averages
            angles_to_head = 180 / np.pi * np.arctan2(xs - xs[0], ys - ys[0])

            tail_tip_deflection = np.mean(angles_to_head[-5:]) # or average the last xs, ys?

            self.tail_tracking_circular_history_time[self.tail_tracking_circular_counter] = self.shared.global_timer.value
            self.tail_tracking_circular_history_tail_tip_deflection[self.tail_tracking_circular_counter] = tail_tip_deflection

            # analyze the bouts (look for vigor thresholds)
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
                self.shared.tail_tracking_new_completed_bout_found[self.fish_index].value = 1  # this needs to be reset by the process that cares about this, likely the visual stimulus

            ######
            # Dataset for this frame is complete, share the all the information information with the global world
            self.fish_camera_image[:frame.shape[0] * frame.shape[1]] = frame.flatten()
            self.shared.fish_camera_image_width[self.fish_index].value = frame.shape[0]
            self.shared.fish_camera_image_height[self.fish_index].value = frame.shape[1]
            self.tail_tracking_xs[:len(xs)] = xs
            self.tail_tracking_ys[:len(ys)] = ys

            # advancing the local counter must still happen after data storage
            self.shared.tail_tracking_circular_counter[self.fish_index].value = self.tail_tracking_circular_counter  # copy the circular counter to the displaying task before



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
                                            self.eye_tracking_circular_history_angle_left_eye[self.tail_tracking_circular_counter],
                                            self.eye_tracking_circular_history_angle_right_eye[self.tail_tracking_circular_counter],
                                            self.tail_tracking_circular_history_tail_tip_deflection[self.tail_tracking_circular_counter],
                                            self.tail_tracking_circular_history_tail_tip_deflection_sliding_window_variance[self.tail_tracking_circular_counter],
                                            self.tail_tracking_circular_history_tail_tip_deflection_sliding_window_mean[self.tail_tracking_circular_counter],
                                            self.tail_tracking_circular_history_bout_information[self.tail_tracking_circular_counter]]) # 0: outside bout, 1: bout start, 2: inside bout, 3: bout stop

                self.tail_shape_data.append([xs, ys])

                if self.shared.experiment_configuration_store_head_tail_movie.value == 1: #storing the tail frames produces massive datasets!
                    self.head_tail_movie.append(frame)


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
                    '''
                    np.savez_compressed(filename,
                                        camera_time=head_tail_data[:, 0],
                                        left_eye_angle=head_tail_data[:, 1],
                                        right_eye_angle=head_tail_data[:, 2],
                                        tail_tip_deflection=head_tail_data[:, 3],
                                        tail_vigor=head_tail_data[:, 4],
                                        tail_bout_information=head_tail_data[:, 5].astype(np.uint),
                                        tail_shape_xs=tail_shape_data[:, 0, :].astype(np.uint16),
                                        tail_shape_ys=tail_shape_data[:, 1, :].astype(np.uint16))'''


                if self.shared.experiment_configuration_store_head_tail_movie.value == 1:

                    filename = os.path.join(rawdata_path, "trial{:03d}_head_tail_movie.mp4".format(self.shared.experiment_flow_control_current_trial.value))

                    '''writer = imageio.get_writer(filename, fps=60)

                    for i in range(len(self.head_tail_data)):
                        # draw the points
                        frame = self.head_tail_movie[i]

                        for j in range(len(self.tail_shape_data[i][0])):

                            cv2.circle(frame, (int(self.tail_shape_data[i][0][j]), int(self.tail_shape_data[i][1][j])), 2, thickness=-1, color=255)
                            cv2.putText(frame, "Time: {:.3f} s".format(self.head_tail_data[i][0]), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=255)

                        writer.append_data(frame)

                    writer.close()'''

                # free the memory
                self.head_tail_movie = []
                self.head_tail_data = []
                self.tail_shape_data = []

                self.shared.experiment_flow_control_currently_storing_head_tail_data[self.fish_index].value = 0
                self.shared.experiment_flow_control_store_head_tail_data_completed[self.fish_index].value = 1


            if self.tail_tracking_circular_counter % 10 == 9:

                self.shared.fish_camera_fps[self.fish_index].value = 10. / (self.shared.fish_camera_timestamp[self.fish_index].value - self.fps_timer_start)

                self.fps_timer_start = self.shared.fish_camera_timestamp[self.fish_index].value

            self.tail_tracking_circular_counter += 1
            if self.tail_tracking_circular_counter == 12000:
                self.tail_tracking_circular_counter = 0

        self.camera_library.close_cam()

        self.camera_library.cleanup()
