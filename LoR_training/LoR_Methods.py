import numpy as np
import glob
import os
import sys

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from Statistical_Methods import Statistic

class Subtrial:
    def __init__(self, stimulus_matrix, HT_matrix, fish_number, trial_number, stimulus_index):
        self.stimulus_matrix = stimulus_matrix
        self.HT_matrix = HT_matrix
        self.fish_number = fish_number
        self.trial_number = trial_number
        self.stimulus_index = stimulus_index
        
        self.bout_start_series = []
        self.bout_end_series = []
        
        self.CL_start_series = [] # 5 elements
        self.CL_end_series = []
        self.Condition_start_series = []
        self.Condition_end_series = []
        
        self.CL_int_angles = []
        self.Condition_int_angles = []
        
        self.CL_preference = [] # organized by 5 sub-subtrials
        self.Condition_preference = []
        
        self.analysis_status = 0
        
    def _bout_detection(self):
        fish_ac_series = Statistic.moving_variance_padded(self.stimulus_matrix['stimulus_data'][:, 10])
        frame_number = 0
        bout_status = 0
        
        for fish_ac in fish_ac_series:
            if (fish_ac >= 0.5) & (bout_status == 0):
                bout_status = 1
                self.bout_start_series.append(frame_number)
            elif (fish_ac < 0.5) & (bout_status == 1):
                bout_status = 0
                self.bout_end_series.append(frame_number)
            
            frame_number += 1
        
    def _stage_segmentation(self):
        stiname = self.stimulus_matrix['stimulus_data'][:, 9]
        frame_number = 0
        prename = 0
        
        for name in stiname:
            if (prename == 0) & (name == 1):
                self.CL_start_series.append(frame_number)
            elif (prename == 1) & ((name == 2)|(name == 3)|(name == 4)):
                self.CL_end_series.append(frame_number - 1)
                self.Condition_start_series.append(frame_number)
            elif ((prename == 2)|(prename == 3)|(prename == 4)) & (name == 0):
                self.Condition_end_series.append(frame_number - 1)
                
                
            frame_number += 1
            prename = name
            
        
    
    def preference_analysis(self):
        if self.analysis_status == 0:
            self._bout_detection()
            self._stage_segmentation()
            
            bout_number = len(self.bout_start_series)
            print(f"Total bout number of fish {self.fish_number}, trial {self.trial_number}: {bout_number} (stimulus_index = {self.stimulus_index})")
            if bout_number == len(self.bout_end_series):
                pass
            elif bout_number == (len(self.bout_end_series) + 1):
                bout_number -= 1
            else:
                raise ValueError("Invalid bout number detection.")
            
            current_bout_number = 0
            
            # print("CL_end_series length:", len(self.CL_end_series))
            # print("bout_start_seriess length:", len(self.bout_start_series))
            
            for i in range(5):
                # print("Sub-subtrial Number:", i)
                CL_left_bout_number = 0
                CL_right_bout_number = 0
                Condition_left_bout_number = 0
                Condition_right_bout_number = 0
                
                CL_int_angle = []
                Condition_int_angle = []
                
                close_loop_preference = []
                condition_group_preference = []
                
                if current_bout_number != bout_number:
                    while self.bout_start_series[current_bout_number] < self.CL_start_series[i]:
                        current_bout_number += 1
                        
                        # print("Current_bout_number in Rest:", current_bout_number)
                
                if current_bout_number != bout_number:
                    # Close Loop Analysis
                    while self.bout_start_series[current_bout_number] < self.CL_end_series[i]:
                        int_time = 0
                        bout_int_frame_number = 0
                        while int_time < 0.1:
                            j = 0
                            int_time += self.stimulus_matrix['stimulus_data'][:, 7][self.bout_start_series[current_bout_number] + j]
                            bout_int_frame_number += 1
                            j += 1
                        
                        int_angle = 0
                        for k in range(bout_int_frame_number):
                            int_angle += self.stimulus_matrix['stimulus_data'][:, 10][self.bout_start_series[current_bout_number] + k]
                        CL_int_angle.append((int_angle / bout_int_frame_number) - self.stimulus_matrix['stimulus_data'][:, 12][self.bout_start_series[current_bout_number]])
                        preference_temp = 0 if ((int_angle / bout_int_frame_number) < self.stimulus_matrix['stimulus_data'][:, 12][self.bout_start_series[current_bout_number]]) else 1
                
                        current_bout_number += 1
                        close_loop_preference.append(preference_temp)
                        
                        # print("Current_bout_number in CL:", current_bout_number)
                        
                        if current_bout_number == bout_number:
                            break
                
                if current_bout_number != bout_number:
                    # Condition Group Analysis
                    while self.bout_start_series[current_bout_number] < self.Condition_end_series[i]:
                        int_time = 0
                        bout_int_frame_number = 0
                        while int_time < 0.1:
                            j = 0
                            int_time += self.stimulus_matrix['stimulus_data'][:, 7][self.bout_start_series[current_bout_number] + j]
                            bout_int_frame_number += 1
                            j += 1
                        
                        int_angle = 0
                        for k in range(bout_int_frame_number):
                            int_angle += self.stimulus_matrix['stimulus_data'][:, 10][self.bout_start_series[current_bout_number] + k]
                        Condition_int_angle.append((int_angle / bout_int_frame_number) - self.stimulus_matrix['stimulus_data'][:, 12][self.bout_start_series[current_bout_number]])
                        preference_temp = 0 if ((int_angle / bout_int_frame_number) < self.stimulus_matrix['stimulus_data'][:, 12][self.bout_start_series[current_bout_number]]) else 1
                        
                        current_bout_number += 1
                        condition_group_preference.append(preference_temp)
                        
                        # print("Current_bout_number in Condition Group:", current_bout_number)
                        
                        if current_bout_number == bout_number:
                            break
                
                
                    
                    
                self.CL_preference.append(close_loop_preference)
                self.Condition_preference.append(condition_group_preference)
                self.CL_int_angles.append(CL_int_angle)
                self.Condition_int_angles.append(Condition_int_angle)
            
            self.analysis_status = 1
        else:
            print("This trial has already been analyzed. Please check the code.")
    
    
    
    def visualization(self, mode,forced=0):
        if (self.analysis_status == 0) & (forced == 0):
            print("\"self.preference_analysis()\" function needs to be executed first. Please check the code.")
        elif (mode == 'stimulus_curr_angle') & ((self.analysis_status == 1) | (forced == 1)):
            ####### Plot using only the stimulus data #######
            time_length = len(self.stimulus_matrix['stimulus_data'][:, 10])
            
            bout_type_series = self.stimulus_matrix['stimulus_data'][:, 15]
            # left:1 ; right:-1 ; indifferent: -0.5 ; start of a bout: 0.5.
            
            bout_color_map = {
                1.0: 'green',         # left
                -1.0: 'red',      # right
                -0.5: 'orange',        # indifferent
                0.5: 'purple',       # start of bout
                0.0: 'black',        # non-bout
            }

            segments_curr_angle = []
            current_type = bout_type_series[0]
            start_idx = 0

            for i in range(1, time_length):
                if bout_type_series[i] != current_type:
                    segments_curr_angle.append({
                        'start': start_idx,
                        'end': i,
                        'color': bout_color_map.get(current_type, 'black')
                    })
                    start_idx = i
                    current_type = bout_type_series[i]

            segments_curr_angle.append({
                'start': start_idx,
                'end': time_length,
                'color': bout_color_map.get(current_type, 'black')
            })
            

            x = np.linspace(0, time_length - 1, time_length)
            y1 = self.stimulus_matrix['stimulus_data'][:, 10]
            y2 = 50 * self.stimulus_matrix['stimulus_data'][:, 3] + 2
            y3 = 50 * 0.05 * np.ones(time_length, dtype=float) + 2
            

            segments_actual_gain = [
                {'start': 0, 'end': time_length - 1, 'color': 'Blue'},
            ]

            segments_full_gain = [
                {'start': 0, 'end': time_length - 1, 'color': 'Gray'},
            ]

            traces = []
            
            left_label_status = 0
            right_label_status = 0
            indifferent_label_status = 0
            start_of_bout_label_status = 0
            non_bout_label_status = 0
            
            
            for seg in segments_curr_angle:
                x_seg = x[seg['start']:seg['end']]
                y_seg = y1[seg['start']:seg['end']]
                
                if (left_label_status == 0) & (seg['color'] == 'green'):
                    left_label_status = 1
                    trace = go.Scatter(
                        x=x_seg,
                        y=y_seg,
                        mode='lines',
                        line=dict(color=seg['color'], width=3),
                        name=f"Left Bouts"
                    )
                elif (right_label_status == 0) & (seg['color'] == 'red'):
                    right_label_status = 1
                    trace = go.Scatter(
                        x=x_seg,
                        y=y_seg,
                        mode='lines',
                        line=dict(color=seg['color'], width=3),
                        name=f"Right Bouts"
                    )
                elif (indifferent_label_status == 0) & (seg['color'] == 'orange'):
                    indifferent_label_status = 1
                    trace = go.Scatter(
                        x=x_seg,
                        y=y_seg,
                        mode='lines',
                        line=dict(color=seg['color'], width=3),
                        name=f"Indifferent Bouts"
                    )
                elif (start_of_bout_label_status == 0) & (seg['color'] == 'purple'):
                    start_of_bout_label_status = 1
                    trace = go.Scatter(
                        x=x_seg,
                        y=y_seg,
                        mode='lines',
                        line=dict(color=seg['color'], width=3),
                        name=f"Start of Bouts"
                    )
                elif (non_bout_label_status == 0) & (seg['color'] == 'black'):
                    non_bout_label_status = 1
                    trace = go.Scatter(
                        x=x_seg,
                        y=y_seg,
                        mode='lines',
                        line=dict(color=seg['color'], width=3),
                        name=f"Outside of Bouts"
                    )
                else:
                    trace = go.Scatter(
                        x=x_seg,
                        y=y_seg,
                        mode='lines',
                        line=dict(color=seg['color'], width=3),
                        showlegend=False
                    )
                traces.append(trace)

            for seg in segments_full_gain:
                x_seg = x[seg['start']:seg['end']]
                y_seg = y3[seg['start']:seg['end']]
                trace = go.Scatter(
                    x=x_seg,
                    y=y_seg,
                    mode='lines',
                    line=dict(color=seg['color'], width=2),
                    name=f"full_gain"
                )
                traces.append(trace)

            for seg in segments_actual_gain:
                x_seg = x[seg['start']:seg['end']]
                y_seg = y2[seg['start']:seg['end']]
                trace = go.Scatter(
                    x=x_seg,
                    y=y_seg,
                    mode='lines',
                    line=dict(color=seg['color'], width=2),
                    name=f"actual_gain"
                )
                traces.append(trace)
    
    
            fig = go.Figure(data=traces)
            fig.update_layout(
                title="tail_angle(stimulus data only)",
                xaxis_title="Time",
                yaxis_title="Value",
                width=1500,
                height=700
            )
            fig.show()
            
        elif (mode == 'stimulus_vigor') & ((self.analysis_status == 1) | (forced == 1)):
            pass
        
        elif (mode == 'HT_curr_angle') & ((self.analysis_status == 1) | (forced == 1)):
            ####### Plot using HT angle data but Stimulus Label #######
            time_length = len(self.stimulus_matrix['stimulus_time'])
            time_length_HT = len(self.HT_matrix['camera_time'])
            x_stimulus = self.stimulus_matrix['stimulus_time'] - self.stimulus_matrix['stimulus_time'][0]
            x_HT = self.HT_matrix['camera_time'] - self.HT_matrix['camera_time'][0]
            y1 = self.HT_matrix['tail_tip_deflection']
            y2 = 50 * self.stimulus_matrix['stimulus_data'][:, 3] + 2
            y3 = 50 * 0.05 * np.ones(time_length, dtype=float) + 2
            y4 = np.ones(time_length, dtype=float) + 10
            
            bout_type_series = self.stimulus_matrix['stimulus_data'][:, 15]
            
            bout_color_map = {
                1.0: 'green',         # left
                -1.0: 'red',      # right
                -0.5: 'orange',        # indifferent
                0.5: 'purple',       # start of bout
                0.0: 'black',        # non-bout
            }
            
            segments_bout_type = []
            current_type = bout_type_series[0]
            start_idx = 0

            for i in range(1, time_length):
                if bout_type_series[i] != current_type:
                    segments_bout_type.append({
                        'start': start_idx,
                        'end': i,
                        'color': bout_color_map.get(current_type, 'black')
                    })
                    start_idx = i
                    current_type = bout_type_series[i]

            segments_bout_type.append({
                'start': start_idx,
                'end': time_length,
                'color': bout_color_map.get(current_type, 'black')
            })
            
            segments_HT_curr_angle = [
                {'start': 0, 'end': time_length_HT - 1, 'color': 'Black'},
            ]
            
            segments_actual_gain = [
                {'start': 0, 'end': time_length - 1, 'color': 'Blue'},
            ]

            segments_full_gain = [
                {'start': 0, 'end': time_length - 1, 'color': 'Gray'},
            ]
            
            

            traces = [] # all curves store here
            
            left_label_status = 0
            right_label_status = 0
            indifferent_label_status = 0
            start_of_bout_label_status = 0
            non_bout_label_status = 0
            
            for seg in segments_bout_type:
                x_seg = x_stimulus[seg['start']:seg['end']]
                y_seg = y4[seg['start']:seg['end']]
                if (left_label_status == 0) & (seg['color'] == 'green'):
                    left_label_status = 1
                    trace = go.Scatter(
                        x=x_seg,
                        y=y_seg,
                        mode='lines',
                        line=dict(color=seg['color'], width=3),
                        name=f"Left Bouts"
                    )
                elif (right_label_status == 0) & (seg['color'] == 'red'):
                    right_label_status = 1
                    trace = go.Scatter(
                        x=x_seg,
                        y=y_seg,
                        mode='lines',
                        line=dict(color=seg['color'], width=3),
                        name=f"Right Bouts"
                    )
                elif (indifferent_label_status == 0) & (seg['color'] == 'orange'):
                    indifferent_label_status = 1
                    trace = go.Scatter(
                        x=x_seg,
                        y=y_seg,
                        mode='lines',
                        line=dict(color=seg['color'], width=3),
                        name=f"Indifferent Bouts"
                    )
                elif (start_of_bout_label_status == 0) & (seg['color'] == 'purple'):
                    start_of_bout_label_status = 1
                    trace = go.Scatter(
                        x=x_seg,
                        y=y_seg,
                        mode='lines',
                        line=dict(color=seg['color'], width=3),
                        name=f"Start of Bouts"
                    )
                elif (non_bout_label_status == 0) & (seg['color'] == 'black'):
                    non_bout_label_status = 1
                    trace = go.Scatter(
                        x=x_seg,
                        y=y_seg,
                        mode='lines',
                        line=dict(color=seg['color'], width=3),
                        name=f"Outside of Bouts"
                    )
                else:
                    trace = go.Scatter(
                        x=x_seg,
                        y=y_seg,
                        mode='lines',
                        line=dict(color=seg['color'], width=3),
                        showlegend=False
                    )
                traces.append(trace)
            
            for seg in segments_HT_curr_angle: # HT_curr_angle curve
                x_seg = x_HT[seg['start']:seg['end']]
                y_seg = y1[seg['start']:seg['end']]
                trace = go.Scatter(
                    x=x_seg,
                    y=y_seg,
                    mode='lines',
                    line=dict(color=seg['color'], width=2),
                    name=f"full_gain"
                )
                traces.append(trace)
            
            
            for seg in segments_full_gain: # full_gain curve
                x_seg = x_stimulus[seg['start']:seg['end']]
                y_seg = y3[seg['start']:seg['end']]
                trace = go.Scatter(
                    x=x_seg,
                    y=y_seg,
                    mode='lines',
                    line=dict(color=seg['color'], width=2),
                    name=f"full_gain"
                )
                traces.append(trace)

            for seg in segments_actual_gain: # actual_gain curve
                x_seg = x_stimulus[seg['start']:seg['end']]
                y_seg = y2[seg['start']:seg['end']]
                trace = go.Scatter(
                    x=x_seg,
                    y=y_seg,
                    mode='lines',
                    line=dict(color=seg['color'], width=2),
                    name=f"actual_gain"
                )
                traces.append(trace)
    
    
            fig = go.Figure(data=traces)
            fig.update_layout(
                title="tail_angle (HT-angle with stimulus-gain)",
                xaxis_title="Time",
                yaxis_title="Value",
                width=1500,
                height=700
            )
            fig.show()
            
            
    def end_of_stage_analysis(self):
        pass

        

class LoR_Analysis:
    @staticmethod
    def check_index(stimulus_index, checkpoint):
        return stimulus_index[checkpoint]
    
class file_process:
    @staticmethod
    def extract_original_folder(rootFolder):
        fish_individuals = sorted(
            os.path.join(rootFolder, fishIndex)
            for fishIndex in os.listdir(rootFolder) 
            if os.path.isdir(os.path.join(rootFolder, fishIndex))
        )
        
        stimulus_data = []
        HT_data = []
        
        for fish_individual in fish_individuals:
            fish_trials_stimulus = sorted(glob.glob(
                os.path.join(fish_individual, "**", "*stimulus_data.npz"),
                recursive=True
            ))
            stimulus_data.append(fish_trials_stimulus)
            
            fish_trials_HT = sorted(glob.glob(
                os.path.join(fish_individual, "**", "*tail_data.npz"),
                recursive=True
            ))
            HT_data.append(fish_trials_HT)
        
        return stimulus_data, HT_data
    
    @staticmethod
    def get_trial_objects(rootFolder):
        