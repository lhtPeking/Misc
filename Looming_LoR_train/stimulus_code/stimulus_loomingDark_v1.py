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
        uniform float condition; // LoR or RoL
        uniform float typeofbout; // correct or incorrect or indifferent
        uniform float time_since_looming; // determines the radius of looming circle
        uniform float time_since_shocking;
        
        in vec2 texcoord;
        out vec4 gl_FragColor;

        #define PI 3.14159265359

        float circle(vec2 st, vec2 center, float radius){
            vec2 dist = st-center;
            return step(radius,
                                dot(dist,dist)*4.0);
        }

        void main(void){
            vec2 st = vec2(texcoord.x, texcoord.y) - vec2(0.5);
            vec3 bg = vec3(1.0)
            
            // Make black outside certain radius
            color = mix(color, vec3(0.0), circle(st,vec2(0.0),1.0));
            // Loom Area
            color = mix(vec3(0.0), color, circle(st,vec2(0.0),time_since_looming*0.0002));
            // Shock Area
            if (0 < time_since_shocking < 0.25) {
                color = vec3(1.0);
            } else {
                color = mix(vec3(0.0), color, circle(st,vec2(0.0),0.01));
            }
            
            float ind_size = 0.002;
            // bout indicator
            vec2 bout_indicator_center = vec2(0.0,-0.5);
            // correct: green
            if (typeofbout == 1.0) {color = mix(vec3(0.0,1.0,0.0), color, circle(st,bout_indicator_center,ind_size));
            // incorrect: red
            } else if (typeofbout == -1.0) {color = mix(vec3(1.0,0.0,0.0), color, circle(st,bout_indicator_center,ind_size));
            // indeterminate: yellow
            } else if (typeofbout == -0.5) {color = mix(vec3(1.0,1.0,0.0), color, circle(st,bout_indicator_center,ind_size));
            // undecided: blue
            } else if (typeofbout == 0.5) {color = mix(vec3(0.0,0.0,1.0), color, circle(st,bout_indicator_center,ind_size));
            // no bout
            } else {color = mix(vec3(0.5), color, circle(st,bout_indicator_center,ind_size));
            }
            
            // condition indicator
            vec2 condition_indicator_center = vec2(0.0,-0.45);
            // Left: purple
            if (condition == 0.0) {color = mix(vec3(0.5,0.0,1.0), color, circle(st,condition_indicator_center,ind_size));
            // Right: orange
            } else if (condition == 1.0) {color = mix(vec3(1.0,0.5,0.0), color, circle(st,condition_indicator_center,ind_size));
            // other
            } else {color = mix(vec3(0.5), color, circle(st,condition_indicator_center,ind_size));
            }
            
            gl_FragColor = vec4(color, 1.0);
        }
    """
]

class MyApp(Panda3D_Scene):
    def __init__(self, shared):

        self.stimulus_number_of_stimuli = 1
        self.stimulus_subtrialpertrial = 1
        self.stimulus_time_per_subtrial = 1200 # 20 mins
        self.stimulus_time_per_stimulus = self.stimulus_time_per_subtrial*self.stimulus_subtrialpertrial
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
            self.cardnodes[fish_index].setShaderInput("typeofbout", 0.0)
            self.cardnodes[fish_index].setShaderInput("condition", 0.0)
            self.cardnodes[fish_index].setShaderInput("time_since_looming", 0.0)
            self.cardnodes[fish_index].setShaderInput("time_since_shocking", 0.0)

        self.baseline_angle = [0 for _ in range(4)]
        self.prev_ac = [0 for _ in range(4)]
        
        self.frame_count_after_start = [1 for _ in range(4)]
        self.time_after_start = [0 for _ in range(4)]
        self.int_angle = [0 for _ in range(4)]
        self.int_vigor = [0 for _ in range(4)]
        self.decided = [0 for _ in range(4)]
        self.decided_prev = [0 for _ in range(4)]
        self.typeofbout = [0 for _ in range(4)] #0: no bout; -0.5: indeterminate; 0.5: too early to decide; 1: correct side; -1: incorrect side
        self.condition = [0 for _ in range(4)]

        self.numcorrectbouts = [0 for _ in range(4)]
        self.numincorrectbouts = [0 for _ in range(4)]
        self.numindetbouts = [0 for _ in range(4)]
        self.numfailed = [0 for _ in range(4)]
        self.time_since_looming = [0 for _ in range(4)]
        self.time_since_normal = [0 for _ in range(4)]
        self.time_since_shocking = [0 for _ in range(4)]
        
        self.isLooming = [0 for _ in range(4)]
        self.isShocking = [0 for _ in range(4)]


    def init_stimulus(self, fish_index, stimulus_index):
        pass


    def update_stimulus(self, fish_index, stimulus_index, stimulus_time, dt):

        if stimulus_time > self.stimulus_time_per_stimulus:
            self.shared.stimulus_flow_control_result_info[fish_index].value = 0
            self.shared.stimulus_flow_control_next_stimulus_requested[fish_index].value = 1

            return None
        

        trial_num = self.shared.experiment_flow_control_current_trial.value
        self.condition[fish_index] = np.floor(trial_num/2) % 2 # 0 is LoR; 1 is RoL

        if (self.isLooming[fish_index] == 0) & (self.isShocking[fish_index] == 0): # Normal
            self.time_since_normal[fish_index] += dt
            self.time_since_looming[fish_index] = 0
            self.time_since_shocking[fish_index] = 0
        elif (self.isLooming[fish_index] == 1) & (self.isShocking[fish_index] == 0): # Looming
            self.time_since_normal[fish_index] = 0
            self.time_since_looming[fish_index] += dt
            self.time_since_shocking[fish_index] = 0
        elif self.isShocking[fish_index] == 1: # Shocking
            self.time_since_normal[fish_index] = 0
            self.time_since_looming[fish_index] = 0
            self.time_since_shocking[fish_index] += dt
            

        fish_ac = np.sqrt(self.shared.tail_tracking_circular_history_tail_tip_deflection_sliding_window_variance[fish_index][self.shared.tail_tracking_circular_counter[fish_index].value])
        curr_angle = self.shared.tail_tracking_circular_history_tail_tip_deflection[fish_index][self.shared.tail_tracking_circular_counter[fish_index].value]
        curr_angle_mean = self.shared.tail_tracking_circular_history_tail_tip_deflection_sliding_window_mean[fish_index][self.shared.tail_tracking_circular_counter[fish_index].value]
        if (fish_ac < 0.2) & (self.prev_ac[fish_index] == 0):
            self.baseline_angle[fish_index] = np.mean([curr_angle_mean,self.baseline_angle[fish_index]])
            self.decided[fish_index] = 0
            self.typeofbout[fish_index] = 0

        if fish_ac < 0.5:
            fish_ac = 0

        
        decisiontime = 0.05
        decisionthreshold = 0.5
        
        ## bout
        if (fish_ac > 0):
            ## start of bout
            if (self.prev_ac[fish_index] == 0): 
                self.typeofbout[fish_index] = 0.5
                self.frame_count_after_start[fish_index] = 1 # reset
                self.time_after_start[fish_index] = dt # reset
                self.int_angle[fish_index] = (curr_angle - self.baseline_angle[fish_index])*dt # reset
                self.decided[fish_index] = 0.5

            ## within bout
            else:
                self.frame_count_after_start[fish_index] += 1
                self.time_after_start[fish_index] += dt
                self.int_angle[fish_index] += (curr_angle - self.baseline_angle[fish_index])*dt

                if ((self.decided[fish_index] == 0) & (np.abs(self.int_angle[fish_index]) / self.time_after_start[fish_index] > decisionthreshold)):
                    self.decided[fish_index] = 1

                    if (self.int_angle[fish_index] / self.time_after_start[fish_index] > decisionthreshold):
                        self.typeofbout[fish_index] = -1 * (-1) ** self.condition[fish_index]
                    elif (self.int_angle[fish_index] / self.time_after_start[fish_index] < -decisionthreshold):
                        self.typeofbout[fish_index] = 1 * (-1) ** self.condition[fish_index]
                    else: self.typeofbout[fish_index] = -0.5

        max_time_looming = 10
        max_time_normal = 30
        
        max_time_shocking = 2

        if self.isLooming[fish_index] == 1: ## currently looming
            if ((self.decided[fish_index] == 1) & (self.decided_prev[fish_index] == 0.5)): # bout occurs
                if self.typeofbout[fish_index] == 1: # correct
                    self.time_since_normal[fish_index] = 0
                    self.isLooming[fish_index] = 0
                    self.numcorrectbouts[fish_index] += 1
                elif self.typeofbout[fish_index] == -1: # incorrect
                    self.isLooming[fish_index] = 0
                    self.isShocking[fish_index] = 1
                    self.numincorrectbouts[fish_index] += 1
                else: self.numindetbouts[fish_index] += 1 # indifferent

            elif self.time_since_looming[fish_index] > max_time_looming: # does't bout til looming ends
                self.time_since_normal[fish_index] = 0
                self.isLooming[fish_index] = 0
                self.numfailed[fish_index] += 1
        elif (self.isLooming[fish_index] == 0) & (self.isShocking[fish_index] == 0): ## currently normal
            if (self.time_since_normal[fish_index] > max_time_normal):
                self.isLooming[fish_index] = 1
                # self.time_since_looming[fish_index] = 0
        elif self.isShocking[fish_index] == 1:
            if self.time_since_shocking[fish_index] > max_time_shocking:
                self.isShocking[fish_index] = 0
        else raise ValueError("Invalid state. It should be \"Normal\" or \"Looming\" or \"Shocking\".")

        fish_speed = fish_ac
        self.prev_ac[fish_index] = fish_speed

        self.decided_prev[fish_index] = self.decided[fish_index]


        self.cardnodes[fish_index].setShaderInput("typeofbout", self.typeofbout[fish_index])
        self.cardnodes[fish_index].setShaderInput("condition", self.condition[fish_index])
        self.cardnodes[fish_index].setShaderInput("time_since_looming", self.time_since_looming[fish_index])
        self.cardnodes[fish_index].setShaderInput("time_since_shocking", self.time_since_shocking[fish_index])

        return [trial_num, stimulus_index, fish_speed, dt, stimulus_time, curr_angle, self.baseline_angle[fish_index], 
                self.frame_count_after_start[fish_index], self.int_angle[fish_index], self.typeofbout[fish_index], 
                self.numcorrectbouts[fish_index], self.numincorrectbouts[fish_index], self.numindetbouts[fish_index], 
                self.numfailed[fish_index], self.isLooming[fish_index], self.time_since_looming[fish_index], 
                self.time_since_normal[fish_index], self.isShocking[fish_index], self.time_since_shocking[fish_index]
                ]