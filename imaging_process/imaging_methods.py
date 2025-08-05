import numpy as np

'''
    An illustration of data structure:
    dff.shape = [neuron, slow_frame]
    swims1.shape = [fast_frame]
    swims2.shape = [fast_frame]
    gain.shape = [fast_frame]
    grating_speed.shape = [fast_frame]
    frames.shape = [fast_frame]
'''

class Imaging:
    def __init__(self, dff_path, frames_path):
        self.dff_path = dff_path
        self.frames_path = frames_path
    
    def rastermap_visualization(self):
        '''
        A visualization method utilizing clustering -> cross-correlation sorting -> upsampling to preserve the structure between superneurons.
        '''
        pass
    
class Behavior:
    def __init__(self, swims1_path, swims2_path, gain_path, grating_speed_path):
        self.swims1_path = swims1_path
        self.swims2_path = swims2_path
        self.gain_path = gain_path
        self.grating_speed_path = grating_speed_path
        
        self.fast_frame_num = len(np.load(self.gain_path))
        
    def ephy_visualization(self):
        '''
        use both swims1 and swims2 data to visualize
        '''
        pass
    
    

class Experiment:
    def __init__(self, imaging, behavior):
        self.imaging = imaging
        self.behavior = behavior
        
    def conjoint_analysis(self):
        pass