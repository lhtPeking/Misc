from panda3d.core import *
import numpy as np
from panda3d_scene import Panda3D_Scene
from scipy.ndimage import gaussian_filter

general_shader = [
    """#version 140

        uniform mat4 p3d_ModelViewProjectionMatrix;
        in vec4 p3d_Vertex;
        in vec2 p3d_MultiTexCoord0;
        out vec2 texcoord;

        void main() {
          gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
          texcoord = p3d_MultiTexCoord0;
        }
    """,

    """#version 140
        in vec2 texcoord;
        out vec4 gl_FragColor;
        uniform float x_offset;
        uniform float y_offset;
        uniform float angle_offset;
        uniform float flicker_seed;
        uniform float stim_type;

        #ifdef GL_ES
        precision mediump float;
        #endif

        #define PI 3.14159265359

        float random (vec2 st) {
            float r = fract(sin(dot(st.xy,
                                vec2(12.31,5.12)))*
                161001.170402);
            float r_binary = 0.0;
            if (r > 0.5){r_binary = 1.0;}
            return r_binary;
        }

        float foreground (vec2 st) {
            float r = fract(sin(dot(st.xy,
                                vec2(10.01,4.02)))*961231.930512);
            float r_binary = 0.0;
            if (r > 0.66){r_binary = 1.0;}
            return r_binary;
        }

        float circle(vec2 st, float radius){
            vec2 dist = st-vec2(0.5);
            return step(radius,
                                dot(dist,dist)*4.0);
        }

        float wedge(vec2 pos, float mid, float halfwidth){
            float ang = atan(pos.y,pos.x)*180.0/PI;
            return step(halfwidth,
                                abs(ang-mid));
        }
        mat2 rotate2d(float angle){
            return mat2(cos(angle),-sin(angle),
                        sin(angle),cos(angle));
        }
        float rightorleft(vec2 loc, float thresh){
	        return step(thresh,loc.x);
        }

        void main() {
            vec2 st = vec2(texcoord.x-x_offset, texcoord.y-y_offset);
            
            // Pattern
            vec2 pos = st-vec2(0.5); 
            vec2 pos_hflip = st-vec2(0.5);
            vec2 pos_stationary = st-vec2(0.5); 
            pos *= rotate2d(angle_offset);
            pos_hflip *= rotate2d(-angle_offset);
                
            
            // Make background
            vec2 radialpos = vec2(length(pos)*33.0,atan(pos.y,pos.x)/(2.0*PI)*51.0);
            vec2 radialpos_hflip = vec2(length(pos_hflip)*33.0,atan(pos_hflip.y,-pos_hflip.x)/(2.0*PI)*51.0);
    
            vec2 i_radialpos = floor(radialpos);
            vec2 f_radialpos = fract(radialpos);
            vec2 i_radialpos_hflip = floor(radialpos_hflip);
            // Assign a random value based on the integer coord
            vec3 color = vec3(random( i_radialpos )); 
            vec3 color_hflip = vec3(random( i_radialpos_hflip ));
            color = mix(color_hflip,color,rightorleft(pos_stationary, 0.0));
            // Uncomment to see the subdivided grid
            //color = vec3(f_radialpos,0.0);
            

            // Make foreground
            vec2 radialpos_foreground_big = vec2(length(pos_stationary)*11.0,atan(pos_stationary.y,pos_stationary.x)/(2.0*PI)*17.0);
            vec2 i_radialpos_foreground_big = floor(radialpos_foreground_big);
            vec2 f_radialpos_foreground_big = fract(radialpos_foreground_big);
            vec2 radialpos_foreground_small = vec2(length(pos_stationary)*33.0,atan(pos_stationary.y,pos_stationary.x)/(2.0*PI)*51.0);
            vec2 i_radialpos_foreground_small = floor(radialpos_foreground_small);
            vec2 f_radialpos_foreground_small = fract(radialpos_foreground_small);
            
            vec3 uniform_foreground = vec3(0.5);
            vec3 stationary_foreground = vec3(random( i_radialpos_foreground_small ));
            vec3 flicker_foreground = vec3(random( i_radialpos_foreground_small*flicker_seed));
            if (stim_type == 0.0){color = mix(uniform_foreground, color, foreground(i_radialpos_foreground_big));
            } else if (stim_type == 1.0){color = mix(stationary_foreground, color, foreground(i_radialpos_foreground_big));
            } else if (stim_type == 2.0){color = mix(flicker_foreground, color, foreground(i_radialpos_foreground_big)); 
            }

            //color = vec3(f_radialpos_foreground,0.0);
            
            // Make grey in center
            color = mix(vec3(0.5), color, circle(st,0.01));
            
            // Make grey behind
            color = mix(vec3(0.5), color, wedge(pos_stationary,-90.0,20.0));
            
            // Make black outside certain radius
            color = mix(color, vec3(0.0), circle(st,1.0));

            gl_FragColor = vec4(color,1.0);
        }
    """
]

class MyApp(Panda3D_Scene):
    def __init__(self, shared):

        self.stimulus_number_of_stimuli = 1
        self.stimulus_time_per_stimulus = 55
        self.timesincebout_thresh = 3

        Panda3D_Scene.__init__(self, shared)

        ############
        # Compile the motion shader
        self.compiled_general_shader = Shader.make(Shader.SLGLSL, general_shader[0], general_shader[1])

        self.x_offset = [0 for _ in range(4)]
        self.y_offset = [0 for _ in range(4)]
        self.angle_offset = [0 for _ in range(4)]
        self.flicker_seed = [1 for _ in range(4)]
        self.stim_type = [0 for _ in range(4)]

        ############ g
        # make the cardnodes for the patterns
        self.make_cardnodes()

        for fish_index in range(4):
            self.cardnodes[fish_index].setShader(self.compiled_general_shader)

            self.cardnodes[fish_index].setShaderInput("x_offset", self.x_offset[fish_index])
            self.cardnodes[fish_index].setShaderInput("y_offset", self.x_offset[fish_index])
            self.cardnodes[fish_index].setShaderInput("angle_offset", self.angle_offset[fish_index])
            self.cardnodes[fish_index].setShaderInput("flicker_seed", self.flicker_seed[fish_index])
            self.cardnodes[fish_index].setShaderInput("stim_type", self.stim_type[fish_index])


        self.prev_gain = [0.1 for _ in range(4)]
        #self.baseline_angle = [0 for _ in range(4)]
        self.prev_ac = [0 for _ in range(4)]
        self.timesincebout = [0 for _ in range(4)]
    
    def init_stimulus(self, fish_index, stimulus_index):
        pass

    def update_stimulus(self, fish_index, stimulus_index, stimulus_time, dt):

        if stimulus_time > self.stimulus_time_per_stimulus:
            self.shared.stimulus_flow_control_result_info[fish_index].value = 0
            self.shared.stimulus_flow_control_next_stimulus_requested[fish_index].value = 1

            return None

        trial_num = self.shared.experiment_flow_control_current_trial.value

        if np.floor(trial_num/10)%3 != self.stim_type[fish_index]: 
            self.stim_type[fish_index] = np.floor(trial_num/10)%3
            self.cardnodes[fish_index].setShaderInput("stim_type", self.stim_type[fish_index])
        if self.stim_type[fish_index] == 2:##flicker
            if np.ceil(stimulus_time*1000/66) != self.flicker_seed[fish_index]:
                self.flicker_seed[fish_index] = np.ceil(stimulus_time*1000/66)
                self.cardnodes[fish_index].setShaderInput("flicker_seed", self.flicker_seed[fish_index])

        fish_ac = np.sqrt(self.shared.tail_tracking_circular_history_tail_tip_deflection_sliding_window_variance[fish_index][self.shared.tail_tracking_circular_counter[fish_index].value])
        curr_angle = self.shared.tail_tracking_circular_history_tail_tip_deflection[fish_index][self.shared.tail_tracking_circular_counter[fish_index].value]
        curr_angle_mean = self.shared.tail_tracking_circular_history_tail_tip_deflection_sliding_window_mean[fish_index][self.shared.tail_tracking_circular_counter[fish_index].value]
        
        if (fish_ac < 0.2) & (self.prev_ac[fish_index] == 0):
            #self.baseline_angle[fish_index] = np.mean([curr_angle_mean,self.baseline_angle[fish_index]])
            self.timesincebout[fish_index] += dt
        else: self.timesincebout[fish_index] = 0
        if fish_ac < 0.5:
            fish_ac = 0


        if 0 <= stimulus_time < 10:  # 0-30 normal; 30-90 normal; 90-150 open loop
            gain = 0.1
            rot_speed = 0
            stimname = 0#'rest'
        elif 10 <= stimulus_time < 55: 
            gain = 0
            rot_speed = 0.5
            stimname = 3#'OL'
        else:
            gain = 0.1
            rot_speed = 0
            stimname = 4#'restother'

        #fish_speed = fish_ac + self.prev_ac*0.5
        fish_speed = fish_ac
        self.prev_ac[fish_index] = fish_speed
        self.prev_gain[fish_index] = gain
#        if fish_speed > 300:
#            fish_speed = 300

        direction = (-1)**(trial_num%2)
        updateamount = rot_speed * direction
        self.angle_offset[fish_index] += updateamount * dt
        self.angle_offset[fish_index] %= 2*np.pi

        self.cardnodes[fish_index].setShaderInput("angle_offset", self.angle_offset[fish_index])

        return [trial_num, gain, rot_speed, fish_speed, self.angle_offset[fish_index], updateamount, dt, stimulus_time, stimname, curr_angle, curr_angle_mean, self.timesincebout[fish_index]]