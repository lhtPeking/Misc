import numpy as np
import glob
import os
import sys

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from Statistical_Methods import Statistic

class Trial:
    def __init__(self, stimulus_matrix, HT_matrix, fish_number, trial_number, condition):
        self.stimulus_matrix = stimulus_matrix
        self.HT_matrix = HT_matrix
        self.fish_number = fish_number
        self.trial_number = trial_number
        self.condition = condition
    
    def visualization(self):
        pass

class file_process:
    @classmethod
    def extract_original_folder(cls, rootFolder):
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
    
    @classmethod
    def get_trial_objects(cls, rootFolder):
        LoR_stimulus_paths, LoR_HT_paths = cls.extract_original_folder(rootFolder)
        
        fish_number = 0
        
        list_condition0 = []
        list_condition1 = []
        
        for stimulus_individual, HT_individual in zip(LoR_stimulus_paths, LoR_HT_paths):
            
            object_condition0 = []
            object_condition1 = []
            
            for stimulus_trial, HT_trial  in zip(stimulus_individual, HT_individual):
                stimulus_matrix = np.load(stimulus_subtrial)
                HT_matrix = np.load(HT_subtrial)
                
                trial_num = stimulus_matrix['stimulus_data'][:, 0][100]
                condition = np.floor(trial_num/2) % 2
                
                trial_object = Trial(stimulus_matrix, HT_matrix, fish_number, trial_num, condition)
                
                if condition == 0:
                    object_condition0.append(trial_object)
                elif condition == 1:
                    object_condition1.append(trial_object)
                else:
                    raise ValueError("Invalid stimulus index.")
                    
                # subtrial_object.preference_analysis()
                subtrial_number += 1
            
            fish_number += 1
            
            list_condition0.append(object_condition0)
            list_condition1.append(object_condition1)
            
        return [list_condition0, list_condition1]