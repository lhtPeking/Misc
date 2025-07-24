import numpy as np
import glob
import os
import sys

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

class stimuli:
    def __init__(self, stimulus_data_paths, HT_data_paths):
        self.stimulus_data_paths = stimulus_data_paths
        self.HT_data_paths = HT_data_paths
        
        self.fish_number = len(stimulus_data_paths)
        self.trial_number_per_fish = len(stimulus_data_paths[0])
        self.stimulus_frame_count = len(np.load(self.stimulus_data_paths[0][0])['stimulus_time'])
        
        self.time_axes = np.load(self.stimulus_data_paths[0][0])['stimulus_time'] - np.load(self.stimulus_data_paths[0][0])['stimulus_time'][0]
    
    def scalable_visualization_contrast(self, fish_index, trial_index):
        stimulus_matrix = np.load(self.stimulus_data_paths[fish_index][trial_index])
        HT_matrix = np.load(self.HT_data_paths[fish_index][trial_index])
        
        time_length_HT = len(HT_matrix['camera_time'])
        x_HT = HT_matrix['camera_time'] - HT_matrix['camera_time'][0]
        y_HT = HT_matrix['tail_tip_deflection']
        
        trace = go.Scatter(
                    x=x_HT,
                    y=y_HT,
                    mode='lines',
                    line=dict(color="blue", width=2),
                    name=f"HT_tail_deflection"
                )
        
        fig = go.Figure(data=trace)
        fig.update_layout(
            title=f"tail_angle from fish{fish_index} trial{trial_index}, contrast={stimulus_matrix['stimulus_data'][0, 13]}",
            xaxis_title="Time",
            yaxis_title="Value",
            width=1500,
            height=700
        )
        
        fig.show()
        
    def struggle_detection(self, threshold=50, mode="stimuli", fish_index=None):
        if mode == "stimuli":
            struggle_vector = np.zeros(self.stimulus_frame_count)
            for fish_index in range(self.fish_number):
                for trial_index in range(self.trial_number_per_fish):
                    stimulus_matrix = np.load(self.stimulus_data_paths[fish_index][trial_index])
                    # HT_matrix = np.load(self.HT_data_paths[fish_index][trial_index])
                    time_axes = stimulus_matrix['stimulus_time'] - stimulus_matrix['stimulus_time'][0]
                    curr_angle = stimulus_matrix['stimulus_data'][:, 9]
                    
                    inbout_series = stimulus_matrix['stimulus_data'][:, 14]
                    diff = np.diff(inbout_series)
                    bout_start_indices = np.where(diff == 1)[0] + 1
                    bout_end_indices = np.where(diff == -1)[0] + 1
                    # experiment start at inbout = 0
                    # print(inbout_series)
                    
                    for bout in range(len(bout_end_indices)):
                        # print("fish_index", fish_index)
                        # print("trial_index", trial_index)
                        # print(bout, bout_start_indices[bout], bout_end_indices[bout])
                        adjust_amount = 0
                        if inbout_series[0] == 1:
                            adjust_amount = 1
                            
                        if bout >= len(bout_start_indices):
                            break
                        
                        curr_max_during_bout = max(np.abs(curr_angle[bout_start_indices[bout]:bout_end_indices[bout+adjust_amount]]))
                        if(curr_max_during_bout >= threshold):
                            struggle_vector[bout_start_indices[bout]] += 1
                            
            num_bins = 150
            bins = np.linspace(self.time_axes.min(), self.time_axes.max(), num_bins + 1)
            counts, bin_edges = np.histogram(self.time_axes, bins=bins, weights=struggle_vector)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            struggle_times = self.time_axes[np.where(struggle_vector == 1)]
            kde = gaussian_kde(struggle_times, bw_method=0.1)
            x_vals = np.linspace(self.time_axes.min(), self.time_axes.max(), 500)
            density = kde(x_vals)

            plt.figure(figsize=(10, 3))
            plt.bar(bin_centers, counts, width=np.diff(bin_edges), align='center', edgecolor='k', alpha=0.4, label='Struggle Count')
            plt.plot(x_vals, density * len(struggle_times) * (bins[1] - bins[0]), color='red', linewidth=2, label='KDE Fit')

            plt.xlabel('Time (s)')
            plt.ylabel('Struggle Count')
            plt.title('Struggle Frequency with KDE Fit')
            plt.legend()
            plt.grid(True, axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()


        elif mode == "fish":
            pass
        else:
            raise ValueError("Invalide Mode. Mode could either be \"stimuli\" or \"fish\".")
        
        return struggle_vector
        
        
        
        
        
        
        
        
        
    def scalable_visualization_coherence(self, fish_index, trial_index):
        stimulus_matrix = np.load(self.stimulus_data_paths[fish_index][trial_index])
        HT_matrix = np.load(self.HT_data_paths[fish_index][trial_index])
        
        time_length_HT = len(HT_matrix['camera_time'])
        x_HT = HT_matrix['camera_time'] - HT_matrix['camera_time'][0]
        y_HT = HT_matrix['tail_tip_deflection']
        
        trace = go.Scatter(
                    x=x_HT,
                    y=y_HT,
                    mode='lines',
                    line=dict(color="blue", width=2),
                    name=f"HT_tail_deflection"
                )
        
        fig = go.Figure(data=trace)
        fig.update_layout(
            title=f"tail_angle from fish{fish_index} trial{trial_index}, coherence={stimulus_matrix['stimulus_data'][0, 13]}",
            xaxis_title="Time",
            yaxis_title="Value",
            width=1500,
            height=700
        )
        
        fig.show()

class file_process:
    @classmethod
    def extract_from_stimuli(cls, rootFolder):
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
        
        # stimulus_data = [[fish0],[fish1],[fish2],[fish3]] = [[trial0, trial1, ...],...], elements are paths.
        return stimulus_data, HT_data