import numpy as np
import glob
import os
import sys

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d
import pywt
import pycwt as wavelet
from scipy.signal import detrend



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
        
    def struggle_detection(self, threshold=50, mode="stimuli"):
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
                            
            # 栅格化原始时间轴
            num_bins = 150
            bins = np.linspace(self.time_axes.min(), self.time_axes.max(), num_bins + 1)
            counts, bin_edges = np.histogram(self.time_axes, bins=bins, weights=struggle_vector)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_width = bins[1] - bins[0]

            # ==== 泊松 λ(t) 估计 ====
            def sliding_poisson_lambda(counts, centers, window_size_bins=5, stride_bins=1):
                lambdas = []
                lambda_times = []
                for i in range(0, len(counts) - window_size_bins + 1, stride_bins):
                    window = counts[i:i+window_size_bins]
                    lam = np.mean(window) / bin_width  # 恢复为“单位时间 λ”
                    center_time = np.mean(centers[i:i+window_size_bins])
                    lambdas.append(lam)
                    lambda_times.append(center_time)
                return np.array(lambda_times), np.array(lambdas)

            # 滑动窗口 λ(t) 曲线
            lambda_times, lambda_vals = sliding_poisson_lambda(counts, bin_centers, window_size_bins=10, stride_bins=1)
            lambda_vals_smooth = gaussian_filter1d(lambda_vals, sigma=2)

            # ==== 绘图 ====
            plt.style.use('dark_background')
            plt.figure(figsize=(10, 3))

            # 原始计数柱状图
            plt.bar(bin_centers, counts, width=np.diff(bin_edges), align='center', edgecolor='white', color='lightgray',
                    alpha=0.4, label='Struggle Count')

            # λ(t) 曲线（泊松滑动估计）
            plt.plot(lambda_times, lambda_vals * bin_width, color='#D76C82', linewidth=2, label='Poisson λ(t)')
            
            # Smoothed λ(t) 曲线（泊松滑动估计）
            plt.plot(lambda_times, lambda_vals_smooth * bin_width, color='#99BC85', linewidth=2, label='Smoothed Poisson λ(t)')

            plt.xlabel('Time (s)')
            plt.ylabel('Struggle Count / Estimated Rate')
            plt.title('Struggle Frequency with Poisson λ(t) Estimation')
            plt.legend()
            plt.grid(True, axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.style.use('dark_background')
            plt.show()
            
            # ==== 输入信号 ====
            signal = lambda_vals_smooth
            t = lambda_times
            dt = np.mean(np.diff(t))  # 时间步长（假设等间距）

            # ==== 选择小波类型和尺度 ====
            wavelet = 'cmor2.0-0.5'  # cmor[bandwidth parameter]-[center frequency]
            scales = np.arange(1, 128)

            # ==== 连续小波变换 ====
            coeffs, freqs = pywt.cwt(signal, scales, wavelet, sampling_period=dt)
            power = np.abs(coeffs) ** 2

            # ==== 可视化 ====
            plt.figure(figsize=(10, 4))
            plt.imshow(power, extent=[t[0], t[-1], freqs[-1], freqs[0]],
                    cmap='jet', aspect='auto', vmin = np.percentile(power, 10), vmax = np.percentile(power, 99.5))
            plt.colorbar(label='Log Power')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title('Wavelet Power Spectrum')
            plt.tight_layout()
            plt.show()


        elif mode == "fish":
            struggle_vector = np.zeros((self.fish_number, self.stimulus_frame_count))

            # ==== 计算每条鱼的 struggle vector ====
            for fish_index in range(self.fish_number):
                for trial_index in range(self.trial_number_per_fish):
                    stimulus_matrix = np.load(self.stimulus_data_paths[fish_index][trial_index])
                    time_axes = stimulus_matrix['stimulus_time'] - stimulus_matrix['stimulus_time'][0]
                    curr_angle = stimulus_matrix['stimulus_data'][:, 9]
                    inbout_series = stimulus_matrix['stimulus_data'][:, 14]
                    diff = np.diff(inbout_series)
                    bout_start_indices = np.where(diff == 1)[0] + 1
                    bout_end_indices = np.where(diff == -1)[0] + 1

                    for bout in range(len(bout_end_indices)):
                        adjust_amount = 0
                        if inbout_series[0] == 1:
                            adjust_amount = 1
                        if bout >= len(bout_start_indices):
                            break
                        if bout + adjust_amount >= len(bout_end_indices):
                            break

                        curr_max_during_bout = max(np.abs(curr_angle[bout_start_indices[bout]:bout_end_indices[bout + adjust_amount]]))
                        if curr_max_during_bout >= threshold:
                            struggle_vector[fish_index, bout_start_indices[bout]] += 1

            # 栅格化时间轴
            num_bins = 150
            bins = np.linspace(self.time_axes.min(), self.time_axes.max(), num_bins + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            bin_width = bins[1] - bins[0]
            
            # ==== 泊松 λ(t) 估计 ====
            def sliding_poisson_lambda(counts, centers, window_size_bins=5, stride_bins=1):
                lambdas = []
                lambda_times = []
                for i in range(0, len(counts) - window_size_bins + 1, stride_bins):
                    window = counts[i:i+window_size_bins]
                    lam = np.mean(window) / bin_width  # 恢复为“单位时间 λ”
                    center_time = np.mean(centers[i:i+window_size_bins])
                    lambdas.append(lam)
                    lambda_times.append(center_time)
                return np.array(lambda_times), np.array(lambdas)

            # 准备画图
            plt.style.use('dark_background')
            plt.figure(figsize=(10, 4))

            lambda_list = []
            t_list = []

            # ==== 对每条鱼单独处理 ====
            for fish_index in range(self.fish_number):
                counts, bin_edges = np.histogram(self.time_axes, bins=bins, weights=struggle_vector[fish_index])

                # 滑动估计泊松 λ(t)
                lambda_times, lambda_vals = sliding_poisson_lambda(counts, bin_centers, window_size_bins=10, stride_bins=1)
                lambda_vals_smooth = gaussian_filter1d(lambda_vals, sigma=2)

                # 绘制 λ(t) 曲线
                plt.plot(lambda_times, lambda_vals_smooth * bin_width, linewidth=2, alpha=0.7, label=f'Fish {fish_index + 1}')

                lambda_list.append(lambda_vals_smooth)
                t_list.append(lambda_times)

            plt.xlabel('Time (s)')
            plt.ylabel('Smoothed λ(t)')
            plt.title('Per-Fish Smoothed Poisson λ(t)')
            plt.legend()
            plt.grid(True, axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()

            # ==== 小波分析：每条鱼单独做 Wavelet → 最后求平均 ====
            wavelet = 'cmor2.0-0.5'
            scales = np.arange(1, 128)

            all_power = []

            for i in range(self.fish_number):
                signal = lambda_list[i]
                t = t_list[i]
                dt = np.mean(np.diff(t))
                coeffs, freqs = pywt.cwt(signal, scales, wavelet, sampling_period=dt)
                power = np.abs(coeffs) ** 2
                all_power.append(power)

            # 对所有鱼的 power 平均
            avg_power = np.mean(all_power, axis=0)

            # ==== 可视化平均 Wavelet Power ====
            plt.figure(figsize=(10, 4))
            plt.imshow(avg_power, extent=[t[0], t[-1], freqs[-1], freqs[0]],
                    cmap='jet', aspect='auto',
                    vmin=np.percentile(avg_power, 10), vmax=np.percentile(avg_power, 99.5))
            plt.colorbar(label='Avg Log Power')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title('Average Wavelet Power Spectrum (All Fish)')
            plt.tight_layout()
            plt.show()
            
        else:
            raise ValueError("Invalide Mode. Mode could either be \"stimuli\" or \"fish\".")
        
        return struggle_vector, self.time_axes, avg_power
        

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