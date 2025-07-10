from panda3d.core import *
import numpy as np
from panda3d_scene import Panda3D_Scene
from scipy.ndimage import gaussian_filter

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
        uniform float iscorrectbout;

        in vec2 texcoord;

        out vec4 gl_FragColor;

        #define PI 3.14159265359

        float circle(vec2 st, vec2 center, float radius){
            vec2 dist = st-center;
            return step(radius,
                                dot(dist,dist)*4.0);
        }

        void main(void){
            float x_ = (2*texcoord.x-x-1)*cos(-pattern_orientation*3.1415/180) - (2*texcoord.y-y-1)*sin(-pattern_orientation*3.1415/180);
            float y_ = (2*texcoord.x-x-1)*sin(-pattern_orientation*3.1415/180) + (2*texcoord.y-y-1)*cos(-pattern_orientation*3.1415/180);

            float r = sqrt(pow(2*texcoord.x-1, 2) + pow(2*texcoord.y-1, 2));
            float c;

            if (r > 1) c = 0;
            else c = 0.5*(sin((x_ - offset)*2*3.1415/wavelength)*contrast+1.0);
            vec3 color = vec3(c);
            
            vec2 st = vec2(texcoord.x, texcoord.y)-vec2(0.5);
            vec2 indicator_center = vec2(0.0,-0.5);
            if (iscorrectbout == 1.0) {color = mix(vec3(0.0,1.0,0.0), color, circle(st,indicator_center,0.005));
            } else if (iscorrectbout == -1.0) {color = mix(vec3(1.0,0.0,0.0), color, circle(st,indicator_center,0.005));
            } else if (iscorrectbout == 0.5) {color = mix(vec3(0.0,0.0,1.0), color, circle(st,indicator_center,0.005));
            } else {color = mix(vec3(0.5), color, circle(st,indicator_center,0.005));
            }
            
            gl_FragColor = vec4(color, 1.0);
        }
    """
]

class MyApp(Panda3D_Scene):
    def __init__(self, shared):

        self.stimulus_number_of_stimuli = 3
        self.stimulus_time_per_stimulus = 120 #180 #240
        #self.stimulus_time_per_subtrial = 120
        #self.stimulus_number_before_switch = 10
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

            self.cardnodes[fish_index].setShaderInput("contrast", 5)
            self.cardnodes[fish_index].setShaderInput("wavelength", 0.3)
            self.cardnodes[fish_index].setShaderInput("x", 0.0)
            self.cardnodes[fish_index].setShaderInput("y", 0.0)
            self.cardnodes[fish_index].setShaderInput("pattern_orientation", 90.0)
            self.cardnodes[fish_index].setShaderInput("offset", 0)
            self.cardnodes[fish_index].setShaderInput("iscorrectbout", 0.0)

        self.pattern_offset = [0 for _ in range(4)]
        self.prev_gain = [0.1 for _ in range(4)]
        self.baseline_angle = [0 for _ in range(4)]
        self.prev_ac = [0 for _ in range(4)]
        self.subtrial_num = [0 for _ in range(4)]
        
        self.frame_count_after_start = [1 for _ in range(4)]
        self.time_after_start = [0 for _ in range(4)]
        self.int_angle = [0 for _ in range(4)]
        self.decided = [0 for _ in range(4)]
        self.iscorrectbout = [0 for _ in range(4)]


    def init_stimulus(self, fish_index, stimulus_index):
        pass


    def update_stimulus(self, fish_index, stimulus_index, stimulus_time, dt):

        if stimulus_time > self.stimulus_time_per_stimulus:
            self.subtrial_num[fish_index] += 1
            self.shared.stimulus_flow_control_result_info[fish_index].value = 0
            self.shared.stimulus_flow_control_next_stimulus_requested[fish_index].value = 1

            return None

        trial_num = self.shared.experiment_flow_control_current_trial.value
        trial_num_processed = self.subtrial_num[fish_index]

        fish_ac = np.sqrt(self.shared.tail_tracking_circular_history_tail_tip_deflection_sliding_window_variance[fish_index][self.shared.tail_tracking_circular_counter[fish_index].value])
        curr_angle = self.shared.tail_tracking_circular_history_tail_tip_deflection[fish_index][self.shared.tail_tracking_circular_counter[fish_index].value]
        curr_angle_mean = self.shared.tail_tracking_circular_history_tail_tip_deflection_sliding_window_mean[fish_index][self.shared.tail_tracking_circular_counter[fish_index].value]
        if (fish_ac < 0.2) & (self.prev_ac[fish_index] == 0):
            self.baseline_angle[fish_index] = np.mean([curr_angle_mean,self.baseline_angle[fish_index]])
        if fish_ac < 0.5:
            fish_ac = 0

        fullgain = 0.05
        if 0 <= stimulus_time < 5:  # 0-5 rest; 5-30 CL; 30-120 asymmetric gain
            gain = fullgain
            forward_speed = 0
            stimname = 0#'rest'
        elif 5 <= stimulus_time < 30:
            gain = fullgain
            forward_speed = 0.1
            stimname = 1#'CL'
        elif 30 <= stimulus_time < 120: # condition
            if stimulus_index == 0:
                ##########
                ## Start of bout
                if (fish_ac > 0) & (self.prev_ac[fish_index] == 0): 
                    gain = 0 # reset
                    self.frame_count_after_start[fish_index] = 1 # reset
                    self.time_after_start[fish_index] = dt # reset
                    self.int_angle[fish_index] = curr_angle # reset
                    self.decided[fish_index] = 0
                    self.iscorrectbout[fish_index] = 0.5 ## inside bout, still deciding
                
                ## During bout, decision time: incorrect
                elif ((fish_ac > 0) & (self.decided[fish_index] == 0)) & ((self.time_after_start[fish_index] > 0.05) & (self.int_angle[fish_index]/self.frame_count_after_start[fish_index] > self.baseline_angle[fish_index])):
                    gain = fullgain*0.2
                    self.decided[fish_index] = 1
                    self.iscorrectbout[fish_index] = -1

                ## During bout, decision time: correct
                elif ((fish_ac > 0) & (self.decided[fish_index] == 0)) & ((self.time_after_start[fish_index] > 0.05) & (self.int_angle[fish_index]/self.frame_count_after_start[fish_index] <= self.baseline_angle[fish_index])):
                    gain = fullgain*1.8
                    self.decided[fish_index] = 1
                    self.iscorrectbout[fish_index] = 1

                ## Outside of bout
                elif (fish_ac == 0):
                    gain = 0
                    self.iscorrectbout[fish_index] = 0
                else:
                    gain = self.prev_gain[fish_index]
        
                ## During bout, but after the starting frame
                if (fish_ac > 0) & (self.prev_ac[fish_index] != 0): 
                    self.frame_count_after_start[fish_index] += 1
                    self.time_after_start[fish_index] += dt
                    self.int_angle[fish_index] += curr_angle
                ##########
                self.cardnodes[fish_index].setShaderInput("iscorrectbout", self.iscorrectbout[fish_index])
                
            elif stimulus_index == 1:
                gain = fullgain*0.2
            else:
                gain = fullgain*1.8
            forward_speed = 0.1
            stimname = 2#'rightOLleftCL'
        else:
            gain = fullgain
            forward_speed = 0
            stimname = 4#'restother'


        #fish_speed = fish_ac + self.prev_ac*0.5
        fish_speed = fish_ac
        self.prev_ac[fish_index] = fish_speed
        self.prev_gain[fish_index] = gain
#        if fish_speed > 300:
#            fish_speed = 300

        self.pattern_offset[fish_index] += (forward_speed - fish_speed*gain) * dt

        self.cardnodes[fish_index].setShaderInput("offset", self.pattern_offset[fish_index])


        return [trial_num, trial_num_processed, stimulus_index, gain, forward_speed, fish_speed, self.pattern_offset[fish_index], dt, stimulus_time, stimname, curr_angle, curr_angle_mean, self.baseline_angle[fish_index], self.frame_count_after_start[fish_index], self.int_angle[fish_index], self.iscorrectbout[fish_index]]

