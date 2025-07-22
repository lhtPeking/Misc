class trial:
    def __init__(self):
        pass

class fish_individual:
    def __init__(self):
        pass
    
    def add_trial():
        pass

class stimuli:
   def __init__(self):
       pass
   
   def add_fish_individual():
       pass
   
class data_process:
    @staticmethod
    def process():
        pass

class visualization:
    @staticmethod
    def scalable_figure():
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