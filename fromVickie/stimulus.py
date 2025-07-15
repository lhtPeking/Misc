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
        uniform float x_offset;
        uniform float y_offset;
        uniform float typeofbout;
        uniform float condition;
        uniform float white;

        in vec2 texcoord;

        out vec4 gl_FragColor;

        #define PI 3.14159265359

        float circle(vec2 st, vec2 center, float radius){
            vec2 dist = st-center;
            return step(radius,
                                dot(dist,dist)*4.0);
        }

        void main(void){
            vec2 st = vec2(texcoord.x-x_offset, texcoord.y-y_offset) - vec2(0.5);

            vec3 color = vec3(white);
            
            // Make black outside certain radius
            color = mix(color, vec3(0.0), circle(st,vec2(0.0),1.0));

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
        self.stimulus_time_per_subtrial = 300
        self.stimulus_time_per_stimulus = self.stimulus_time_per_subtrial*self.stimulus_subtrialpertrial #180 #240
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

            #self.cardnodes[fish_index].setShaderInput("contrast", 5)
            #self.cardnodes[fish_index].setShaderInput("wavelength", 0.3)
            self.cardnodes[fish_index].setShaderInput("x_offset", 0.0)
            self.cardnodes[fish_index].setShaderInput("y_offset", 0.0)
            #self.cardnodes[fish_index].setShaderInput("pattern_orientation", 90.0)
            #self.cardnodes[fish_index].setShaderInput("offset", 0)
            self.cardnodes[fish_index].setShaderInput("typeofbout", 0.0)
            self.cardnodes[fish_index].setShaderInput("condition", 0.0)
            self.cardnodes[fish_index].setShaderInput("white", 1.0)

        #self.pattern_offset = [0 for _ in range(4)]
        #self.prev_gain = [0.1 for _ in range(4)]
        self.baseline_angle = [0 for _ in range(4)]
        self.prev_ac = [0 for _ in range(4)]
        #self.subtrial_num = [0 for _ in range(4)]
        
        self.frame_count_after_start = [1 for _ in range(4)]
        self.time_after_start = [0 for _ in range(4)]
        self.int_angle = [0 for _ in range(4)]
        self.int_vigor = [0 for _ in range(4)]
        #self.vigor_compensatory = [0 for _ in range(4)]
        #self.time_compensatory = [0 for _ in range(4)]
        self.decided = [0 for _ in range(4)]
        self.decided_prev = [0 for _ in range(4)]
        self.typeofbout = [0 for _ in range(4)] #0: no bout; -0.5: indeterminate; 0.5: too early to decide; 1: correct side; -1: incorrect side
        self.condition = [0 for _ in range(4)]

        self.time_since_bout_white = [0 for _ in range(4)]
        self.numcorrectbouts = [0 for _ in range(4)]
        self.numincorrectbouts = [0 for _ in range(4)]
        self.numindetbouts = [0 for _ in range(4)]
        self.numfailed = [0 for _ in range(4)]
        self.white = [1 for _ in range(4)]
        self.time_since_black = [0 for _ in range(4)]
        self.time_since_white = [0 for _ in range(4)]
        
        #self.advised_stimulus_index = [-1 for _ in range(4)]
        #self.index_sequence = [np.random.permutation([0, 0, 0, 0, 1, 2]) for _ in range(4)]


    def init_stimulus(self, fish_index, stimulus_index):
        pass


    def update_stimulus(self, fish_index, stimulus_index, stimulus_time, dt):

        if stimulus_time > self.stimulus_time_per_stimulus:
            self.shared.stimulus_flow_control_result_info[fish_index].value = 0
            self.shared.stimulus_flow_control_next_stimulus_requested[fish_index].value = 1

            return None
        
        #self.subtrial_num[fish_index] = np.floor(stimulus_time / self.stimulus_time_per_subtrial)
        trial_num = self.shared.experiment_flow_control_current_trial.value
        #trial_num_processed = self.subtrial_num[fish_index] + trial_num * self.stimulus_subtrialpertrial
        self.condition[fish_index] = np.floor(trial_num/2) % 2 # 0 is LoR; 1 is RoL

        if self.white[fish_index] == 0:
            self.time_since_black[fish_index] += dt
            self.time_since_white[fish_index] = 0
            self.time_since_bout_white[fish_index] = 0
        else:
            self.time_since_white[fish_index] += dt
            self.time_since_black[fish_index] = 0

        fish_ac = np.sqrt(self.shared.tail_tracking_circular_history_tail_tip_deflection_sliding_window_variance[fish_index][self.shared.tail_tracking_circular_counter[fish_index].value])
        curr_angle = self.shared.tail_tracking_circular_history_tail_tip_deflection[fish_index][self.shared.tail_tracking_circular_counter[fish_index].value]
        curr_angle_mean = self.shared.tail_tracking_circular_history_tail_tip_deflection_sliding_window_mean[fish_index][self.shared.tail_tracking_circular_counter[fish_index].value]
        if (fish_ac < 0.2) & (self.prev_ac[fish_index] == 0):
            self.baseline_angle[fish_index] = np.mean([curr_angle_mean,self.baseline_angle[fish_index]])
            self.decided[fish_index] = 0
            #if (self.time_compensatory[fish_index] <= 0):
            self.typeofbout[fish_index] = 0
            if self.white[fish_index] == 1: self.time_since_bout_white[fish_index] += dt
        else: self.time_since_bout_white[fish_index] = 0

        if fish_ac < 0.5:
            fish_ac = 0

        
        decisiontime = 0.05
        decisionthreshold = 0.5
        #fullgain = 0.05
        
        ## bout
        if (fish_ac > 0):
            ## start of bout
            if (self.prev_ac[fish_index] == 0): 
                self.typeofbout[fish_index] = 0.5
                self.frame_count_after_start[fish_index] = 1 # reset
                self.time_after_start[fish_index] = dt # reset
                self.int_angle[fish_index] = (curr_angle - self.baseline_angle[fish_index])*dt # reset
                self.decided[fish_index] = 0.5

                #transgain = 0.1*defaultgain_trans
                #rotgain = defaultgain_rot

            ## within bout
            else:
                self.frame_count_after_start[fish_index] += 1
                self.time_after_start[fish_index] += dt
                self.int_angle[fish_index] += (curr_angle - self.baseline_angle[fish_index])*dt
                ## at decision time
                #curr_angle_abs = np.abs(curr_angle - self.baseline_angle[fish_index])
                #prev_angle_abs = np.abs(self.prev_angle[fish_index] - self.baseline_angle[fish_index])

                if ((self.decided[fish_index] == 0) & (np.abs(self.int_angle[fish_index]) / self.time_after_start[fish_index] > decisionthreshold)):#(self.time_after_start[fish_index] > decisiontime)):
                    #currtailmean = self.int_angle[fish_index]/self.frame_count_after_start[fish_index]
                    self.decided[fish_index] = 1

                    if (self.int_angle[fish_index] / self.time_after_start[fish_index] > decisionthreshold):
                        self.typeofbout[fish_index] = -1 * (-1) ** self.condition[fish_index]
                    elif (self.int_angle[fish_index] / self.time_after_start[fish_index] < -decisionthreshold):
                        self.typeofbout[fish_index] = 1 * (-1) ** self.condition[fish_index]
                    else: self.typeofbout[fish_index] = -0.5

        max_time_black = 10 # go back to white if fish swims correctly or up to 10s in dark
        min_time_white = 5 # at least 5s bright with no swimming, then switch to dark
        max_time_white = 10 # don't want it to only be bright if always swimming...

        if self.white[fish_index] == 0: ## currently black
            if ((self.decided[fish_index] == 1) & (self.decided_prev[fish_index] == 0.5)):
                if self.typeofbout[fish_index] == 1:
                    self.time_since_white[fish_index] = 0
                    self.white[fish_index] = 1
                    self.numcorrectbouts[fish_index] += 1
                elif self.typeofbout[fish_index] == -1:
                    self.numincorrectbouts[fish_index] += 1
                else: self.numindetbouts[fish_index] += 1

            elif self.time_since_black[fish_index] > max_time_black:
                self.time_since_white[fish_index] = 0
                self.white[fish_index] = 1
                self.numfailed[fish_index] += 1
        else: ## currently white
            if (self.time_since_bout_white[fish_index] > min_time_white) | (self.time_since_white[fish_index] > max_time_white):
                self.white[fish_index] = 0
                self.time_since_black[fish_index] = 0


        #fish_speed = fish_ac + self.prev_ac*0.5
        fish_speed = fish_ac
        self.prev_ac[fish_index] = fish_speed
        #self.prev_gain[fish_index] = gain
        self.decided_prev[fish_index] = self.decided[fish_index]
#        if fish_speed > 300:
#            fish_speed = 300

        self.cardnodes[fish_index].setShaderInput("typeofbout", self.typeofbout[fish_index])
        self.cardnodes[fish_index].setShaderInput("condition", self.condition[fish_index])
        self.cardnodes[fish_index].setShaderInput("white", self.white[fish_index])

        return [trial_num, stimulus_index, fish_speed, dt, stimulus_time, curr_angle, self.baseline_angle[fish_index], self.frame_count_after_start[fish_index], self.int_angle[fish_index], self.typeofbout[fish_index], self.numcorrectbouts[fish_index], self.numincorrectbouts[fish_index], self.numindetbouts[fish_index], self.numfailed[fish_index], self.white[fish_index], self.time_since_black[fish_index], self.time_since_white[fish_index]]

