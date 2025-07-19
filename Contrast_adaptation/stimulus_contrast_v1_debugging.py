from panda3d.core import *
import numpy as np
from panda3d_scene import Panda3D_Scene
from scipy.ndimage import gaussian_filter

### For this experiment, trial number should be set to 6.

sine_grating_motion_shader = [
    """ #version 140
        uniform mat4 p3d_ModelViewProjectionMatrix;
        in vec4 p3d_Vertex;
        in vec2 p3d_MultiTexCoord0;

        out vec2 texcoord;

        void main(void) {
           gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
           texcoord = p3d_MultiTexCoord0;
        }
    """,

    """ #version 140
        uniform float contrast;
        uniform float wavelength;
        uniform float x;
        uniform float y;
        uniform float pattern_orientation;
        uniform float offset;
        uniform float display_mode;
        uniform float dynamic_contrast;

        in vec2 texcoord;

        out vec4 gl_FragColor;
        
        float rand(vec2 co) {
            return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
        }

        void main(void) {
            float x_ = (2*texcoord.x-x-1)*cos(-pattern_orientation*3.1415/180) - (2*texcoord.y-y-1)*sin(-pattern_orientation*3.1415/180);
            float y_ = (2*texcoord.x-x-1)*sin(-pattern_orientation*3.1415/180) + (2*texcoord.y-y-1)*cos(-pattern_orientation*3.1415/180);

            float r = sqrt(pow(2*texcoord.x-1, 2) + pow(2*texcoord.y-1, 2));
            float c;
            
            if (r > 1) c = 0;
            else if ((r <= 1) && (display_mode == 0)) c = 0.5*(sin((x_ - offset)*2*3.1415/wavelength)*contrast+1.0);
            else if ((r <= 1) && (display_mode == 1)) { 
                if (c >= 1) c = 1 - dynamic_contrast;
                else if (c <= 0) c = dynamic_contrast;
            }
            
            gl_FragColor = vec4(c, c, c, 1.0);
        }
    """
]

class MyApp(Panda3D_Scene):
    def __init__(self, shared):

        self.stimulus_number_of_stimuli = 1
        self.stimulus_time_per_stimulus = 150
        self.prev_ac = 0
        Panda3D_Scene.__init__(self, shared)

        ############
        # Compile the motion shader
        self.compiled_general_shader = Shader.make(Shader.SLGLSL, sine_grating_motion_shader[0], sine_grating_motion_shader[1])
        self.compiled_sine_grating_motion_shader = Shader.make(Shader.SLGLSL, sine_grating_motion_shader[0],
                                                               sine_grating_motion_shader[1])

        ############
        # make the cardnodes for the patterns
        self.make_cardnodes()

        for fish_index in range(4):
            self.cardnodes[fish_index].setShader(self.compiled_sine_grating_motion_shader)

            self.cardnodes[fish_index].setShaderInput("dynamic_contrast", 5)
            self.cardnodes[fish_index].setShaderInput("wavelength", 0.3)
            self.cardnodes[fish_index].setShaderInput("x", 0.0)
            self.cardnodes[fish_index].setShaderInput("y", 0.0)
            self.cardnodes[fish_index].setShaderInput("pattern_orientation", 90.0)
            self.cardnodes[fish_index].setShaderInput("offset", 0)
            
            ### display mode: 0 for normal, 1 for low-contrast. Change this parameter only during "Condition" stage.
            self.cardnodes[fish_index].setShaderInput("display_mode", 0)
            self.cardnodes[fish_index].setShaderInput("dynamic_contrast", 1.0)

        self.pattern_offset = [0 for _ in range(4)]
        self.prev_gain = [0.1 for _ in range(4)]
        self.baseline_angle = [0 for _ in range(4)]
        self.prev_ac = [0 for _ in range(4)]
        self.pattern_updateamount_CL_history = [[] for _ in range(4)]
        self.pattern_updateamount_counter = [0 for _ in range(4)]
        
        self.dynamic_contrast = [[np.random.permutation([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])] for _ in range(4)]
        self.display_mode = [0 for _ in range(4)]

    def init_stimulus(self, fish_index, stimulus_index):
        pass

    def update_stimulus(self, fish_index, stimulus_index, stimulus_time, dt):

        if stimulus_time > self.stimulus_time_per_stimulus:
            self.shared.stimulus_flow_control_result_info[fish_index].value = 0
            self.shared.stimulus_flow_control_next_stimulus_requested[fish_index].value = 1

            return None

        trial_num = self.shared.experiment_flow_control_current_trial.value
        fish_ac = np.sqrt(self.shared.tail_tracking_circular_history_tail_tip_deflection_sliding_window_variance[fish_index][self.shared.tail_tracking_circular_counter[fish_index].value])
        curr_angle = self.shared.tail_tracking_circular_history_tail_tip_deflection[fish_index][self.shared.tail_tracking_circular_counter[fish_index].value]
        curr_angle_mean = self.shared.tail_tracking_circular_history_tail_tip_deflection_sliding_window_mean[fish_index][self.shared.tail_tracking_circular_counter[fish_index].value]
        
        if (fish_ac < 0.2) & (self.prev_ac[fish_index] == 0):
            self.baseline_angle[fish_index] = np.mean([curr_angle_mean,self.baseline_angle[fish_index]])
        if fish_ac < 0.5:
            fish_ac = 0

        self.display_mode[fish_index] = 0 # reset
        # 0-10 rest; 10-50 normal CL; 50-150 condition;
        if 0 <= stimulus_time < 10: # 'rest'
            gain = 0.1
            forward_speed = 0
            stimname = 0 
        elif 10 <= stimulus_time < 50: # 'CL'
            gain = 0.1
            forward_speed = 0.1
            stimname = 1 
        elif 50 <= stimulus_time < 150: # 'Condition'
            self.display_mode[fish_index] = 1
            gain = 0.1
            forward_speed = 0.1
            stimname = 3 
        else: #'restother'
            gain = 0.1
            forward_speed = 0
            stimname = 4

        fish_speed = fish_ac
        self.prev_ac[fish_index] = fish_speed
        self.prev_gain[fish_index] = gain

        updateamount = forward_speed - fish_speed*gain

        self.pattern_offset[fish_index] += updateamount * dt

        self.cardnodes[fish_index].setShaderInput("offset", self.pattern_offset[fish_index])
        self.cardnodes[fish_index].setShaderInput("display_mode", self.display_mode[fish_index])
        self.cardnodes[fish_index].setShaderInput("dynamic_contrast", self.dynamic_contrast[fish_index][trial_num])


        return [trial_num, gain, forward_speed, fish_speed, self.pattern_offset[fish_index],
                updateamount, dt, stimulus_time, stimname, curr_angle, curr_angle_mean,
                self.baseline_angle[fish_index], self.display_mode[fish_index], self.dynamic_contrast[fish_index]]
