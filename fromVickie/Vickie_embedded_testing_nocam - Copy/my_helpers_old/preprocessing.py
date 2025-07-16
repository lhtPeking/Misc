import numpy as np
import pylab as pl
import os
from scipy.interpolate import interp1d
import scipy.io as sio
import numpy as np
import pandas as pd
from scipy.ndimage.filters import maximum_filter
from my_helpers import helpers


def find_events(data, dt, window, start_threshold, end_threshold, minimal_inter_event_time):

    data_rolling_var = pd.rolling_var(data, window=int(window / dt), center=True)

    event_start_indices = np.where((data_rolling_var[:-1] <= start_threshold) & (data_rolling_var[1:] > start_threshold))[0].tolist()
    event_end_indices = []

    i = 0
    while i < len(event_start_indices):

        ind = np.where(data_rolling_var[event_start_indices[i]:] < end_threshold)[0]

        if len(ind) > 0:
            event_end_indices.append(event_start_indices[i] + ind[0])
            i += 1
        else:
            event_start_indices.pop(i) # if we did not find an end, the beginning is also invalid

    # if interevents are too small, concateate them
    i = 0
    while i < len(event_start_indices)-1:
        #print(i)
        if (event_start_indices[i + 1] - event_end_indices[i])*dt < minimal_inter_event_time:
            event_start_indices.pop(i+1)
            event_end_indices.pop(i)
        else:
            i += 1

    return event_start_indices, event_end_indices

def get_aligned_data_2P_setup(path, z_plane, trial, dt, interpolate_tail_shape=False):

    filename_visual_stimulus_data = "z_plane{:04d}_trial{:03d}_visual_stimulus_data.npz".format(z_plane, trial)
    filename_tail_tracking_data = "z_plane{:04d}_trial{:03d}_tail_data.npz".format(z_plane, trial)
    filename_head_data = "z_plane{:04d}_trial{:03d}_head_data.npz".format(z_plane, trial)

    visual_stimulus_data = np.load(os.path.join(path, filename_visual_stimulus_data))
    tail_data = np.load(os.path.join(path, filename_tail_tracking_data))
    head_data = np.load(os.path.join(path, filename_head_data))

    head_time = head_data["head_time"]
    tail_time = tail_data["tail_time"]

    if "visual_stimulus_time" in visual_stimulus_data.keys():
        visual_stimulus_extra_infos_time = visual_stimulus_data["visual_stimulus_time"]

    #tail_bout_information = tail_data["tail_bout_information"]
    #bout_start_times = tail_time[tail_bout_information == 1]
    #bout_end_times = tail_time[tail_bout_information == 3]

    # TODO: take care of what happens when bouts happen right at the beginning of a trial
    t_start = min(head_time[0], tail_time[0])
    t_end = max(head_time[-1], tail_time[-1])

    experiment_time = np.arange(t_start, t_end, dt)
    #print(head_data.keys())

    if interpolate_tail_shape == True:
        tail_shape_xs = interp1d(tail_time, tail_data["tail_shape_xs"], axis=0, bounds_error=False)(experiment_time)
        tail_shape_ys = interp1d(tail_time, tail_data["tail_shape_ys"], axis=0, bounds_error=False)(experiment_time)
    else:
        tail_shape_xs = None
        tail_shape_ys = None

    tail_tip_deflection = interp1d(tail_time, tail_data["tail_tip_deflection"], bounds_error=False)(experiment_time)

    #tail_vigor = interp1d(tail_time, tail_data["tail_vigor"], bounds_error=False)(experiment_time)

    head_right_eye_x = interp1d(head_time, head_data["head_right_eye_x"], bounds_error=False)(experiment_time)
    head_right_eye_y = interp1d(head_time, head_data["head_right_eye_y"], bounds_error=False)(experiment_time)
    head_right_eye_length = interp1d(head_time, head_data["head_right_eye_length"], bounds_error=False)(experiment_time)
    head_right_eye_width = interp1d(head_time, head_data["head_right_eye_width"], bounds_error=False)(experiment_time)
    head_right_eye_angle = interp1d(head_time, head_data["head_right_eye_angle"], bounds_error=False)(experiment_time)

    head_left_eye_x = interp1d(head_time, head_data["head_left_eye_x"], bounds_error=False)(experiment_time)
    head_left_eye_y = interp1d(head_time, head_data["head_left_eye_y"], bounds_error=False)(experiment_time)
    head_left_eye_length = interp1d(head_time, head_data["head_left_eye_length"], bounds_error=False)(experiment_time)
    head_left_eye_width = interp1d(head_time, head_data["head_left_eye_width"], bounds_error=False)(experiment_time)
    head_left_eye_angle = interp1d(head_time, head_data["head_left_eye_angle"], bounds_error=False)(experiment_time)

    # or feature tracking squares
    head_mouth_square_variance = interp1d(head_time, head_data["head_mouth_square_variance"], bounds_error=False)(experiment_time)
    head_left_side_square_variance = interp1d(head_time, head_data["head_left_side_square_variance"], bounds_error=False)(experiment_time)
    head_right_side_square_variance = interp1d(head_time, head_data["head_right_side_square_variance"], bounds_error=False)(experiment_time)
    head_heart_square_mean = interp1d(head_time, head_data["head_heart_square_mean"], bounds_error=False)(experiment_time)

    # Linear or nearest? TODO


    if "visual_stimulus_time" in visual_stimulus_data.keys():
        visual_stimulus_extra_infos_data = interp1d(visual_stimulus_extra_infos_time, visual_stimulus_data["visual_stimulus_data"], axis=0, bounds_error=False)(experiment_time)
        print('dfgdfg')
    else:
        visual_stimulus_extra_infos_data = None

    f_index_finder = interp1d(experiment_time, range(len(experiment_time)), kind='nearest')

    # don't store that information any more... if this has been used for closed loop, use other features
    #bout_start_indices = f_index_finder(bout_start_times).astype(np.int)
    #bout_end_indices = f_index_finder(bout_end_times).astype(np.int)

    visual_stimulus_start_indices = f_index_finder(visual_stimulus_data["visual_stimulus_start_times"]).astype(np.int)
    visual_stimulus_start_stimulus_numbers = visual_stimulus_data["visual_stimulus_start_indices"]

    head_right_eye_angle_event_start_indices, head_right_eye_angle_event_end_indices = find_events(head_right_eye_angle, dt = dt, window = 0.05, start_threshold = 1, end_threshold = 0.1, minimal_inter_event_time = 0.3)
    head_left_eye_angle_event_start_indices, head_left_eye_angle_event_end_indices = find_events(head_left_eye_angle, dt = dt, window = 0.05, start_threshold = 1, end_threshold = 0.1, minimal_inter_event_time = 0.3)
    tail_tip_deflection_event_start_indices, tail_tip_deflection_event_end_indices = find_events(tail_tip_deflection, dt = dt, window = 0.05, start_threshold = 1, end_threshold = 0.1, minimal_inter_event_time = 0.3)

    # refine the start indices for the tail tip deflection, find a local maximum, or minimum (important for aligning bouts)
    for i in range(len(tail_tip_deflection_event_start_indices)):
        window_diff = np.diff(tail_tip_deflection[int(tail_tip_deflection_event_start_indices[i] - 0.1 / dt):tail_tip_deflection_event_start_indices[i]])

        ind = np.where(((window_diff[:-1] > 0) & (window_diff[1:] < 0)) | ((window_diff[:-1] < 0) & (window_diff[1:] > 0)))[0]
        if len(ind) > 0:
            tail_tip_deflection_event_start_indices[i] = int(tail_tip_deflection_event_start_indices[i] - 0.1 / dt) + ind[-1] + 1

    # get the tail vigor
    tail_vigor = pd.rolling_var(tail_tip_deflection, window=int(0.05 / dt), center=True)

    ################
    # TODO: in the future, also align the 2P data

    all_behavior_data = dict({ "experiment_time": experiment_time,
                               "tail_shape_xs": tail_shape_xs,
                               "tail_shape_ys": tail_shape_ys,
                               "tail_tip_deflection": tail_tip_deflection,
                               "tail_vigor": tail_vigor,
                               "head_right_eye_x": head_right_eye_x,
                               "head_right_eye_y": head_right_eye_y,
                               "head_right_eye_length": head_right_eye_length,
                               "head_right_eye_width": head_right_eye_width,
                               "head_right_eye_angle": head_right_eye_angle,
                               "head_left_eye_x": head_left_eye_x,
                               "head_left_eye_y": head_left_eye_y,
                               "head_left_eye_length": head_left_eye_length,
                               "head_left_eye_width": head_left_eye_width,
                               "head_left_eye_angle": head_left_eye_angle,
                               "head_mouth_square_variance": head_mouth_square_variance,
                               "head_left_side_square_variance": head_left_side_square_variance,
                               "head_right_side_square_variance": head_right_side_square_variance,
                               "head_heart_square_mean": head_heart_square_mean,
                               "visual_stimulus_extra_infos_data": visual_stimulus_extra_infos_data,
                               "head_right_eye_angle_event_start_indices": head_right_eye_angle_event_start_indices,
                               "head_right_eye_angle_event_end_indices": head_right_eye_angle_event_end_indices,
                               "head_left_eye_angle_event_start_indices": head_left_eye_angle_event_start_indices,
                               "head_left_eye_angle_event_end_indices": head_left_eye_angle_event_end_indices,
                               "tail_tip_deflection_event_start_indices": tail_tip_deflection_event_start_indices,
                               "tail_tip_deflection_event_end_indices": tail_tip_deflection_event_end_indices,
                               "visual_stimulus_start_indices": visual_stimulus_start_indices,
                               "visual_stimulus_end_indices": np.r_[visual_stimulus_start_indices[1:], len(experiment_time)],
                               "visual_stimulus_start_stimulus_numbers": visual_stimulus_start_stimulus_numbers
                               })

    return all_behavior_data


def get_aligned_data_2P_setup2(path, z_plane, trial, dt, interpolate_tail_shape=False):
    filename_visual_stimulus_data = "z_plane{:04d}_trial{:03d}_stimulus_data.npz".format(z_plane, trial)
    filename_tail_tracking_data = "z_plane{:04d}_trial{:03d}_tail_data.npz".format(z_plane, trial)
    filename_head_data = "z_plane{:04d}_trial{:03d}_head_data.npz".format(z_plane, trial)

    visual_stimulus_data = np.load(os.path.join(path, filename_visual_stimulus_data))
    tail_data = np.load(os.path.join(path, filename_tail_tracking_data))
    head_data = np.load(os.path.join(path, filename_head_data))


    head_time = head_data["head_time"]
    tail_time = tail_data["tail_time"]

    #if "visual_stimulus_time" in visual_stimulus_data.keys():
    #    visual_stimulus_extra_infos_time = visual_stimulus_data["visual_stimulus_time"]

    # tail_bout_information = tail_data["tail_bout_information"]
    # bout_start_times = tail_time[tail_bout_information == 1]
    # bout_end_times = tail_time[tail_bout_information == 3]

    # TODO: take care of what happens when bouts happen right at the beginning of a trial
    t_start = min(head_time[0], tail_time[0])
    t_end = max(head_time[-1], tail_time[-1])

    experiment_time = np.arange(t_start, t_end, dt)
    # print(head_data.keys())

    if interpolate_tail_shape == True:
        tail_shape_xs = interp1d(tail_time, tail_data["tail_shape_xs"], axis=0, bounds_error=False)(experiment_time)
        tail_shape_ys = interp1d(tail_time, tail_data["tail_shape_ys"], axis=0, bounds_error=False)(experiment_time)
    else:
        tail_shape_xs = None
        tail_shape_ys = None

    tail_tip_deflection = interp1d(tail_time, tail_data["tail_tip_deflection"], bounds_error=False)(experiment_time)


    # tail_vigor = interp1d(tail_time, tail_data["tail_vigor"], bounds_error=False)(experiment_time)

    #head_right_eye_x = interp1d(head_time, head_data["head_right_eye_x"], bounds_error=False)(experiment_time)
    #head_right_eye_y = interp1d(head_time, head_data["head_right_eye_y"], bounds_error=False)(experiment_time)
    #head_right_eye_length = interp1d(head_time, head_data["head_right_eye_length"], bounds_error=False)(experiment_time)
    #head_right_eye_width = interp1d(head_time, head_data["head_right_eye_width"], bounds_error=False)(experiment_time)
    head_right_eye_angle = interp1d(head_time, head_data["head_right_eye_angle"], bounds_error=False)(experiment_time)

    #head_left_eye_x = interp1d(head_time, head_data["head_left_eye_x"], bounds_error=False)(experiment_time)
    #head_left_eye_y = interp1d(head_time, head_data["head_left_eye_y"], bounds_error=False)(experiment_time)
    #head_left_eye_length = interp1d(head_time, head_data["head_left_eye_length"], bounds_error=False)(experiment_time)
    #head_left_eye_width = interp1d(head_time, head_data["head_left_eye_width"], bounds_error=False)(experiment_time)
    head_left_eye_angle = interp1d(head_time, head_data["head_left_eye_angle"], bounds_error=False)(experiment_time)

    # or feature tracking squares
    #head_mouth_square_variance = interp1d(head_time, head_data["head_mouth_square_variance"], bounds_error=False)(experiment_time)
    #head_left_side_square_variance = interp1d(head_time, head_data["head_left_side_square_variance"], bounds_error=False)(experiment_time)
    #head_right_side_square_variance = interp1d(head_time, head_data["head_right_side_square_variance"], bounds_error=False)(experiment_time)
    #head_heart_square_mean = interp1d(head_time, head_data["head_heart_square_mean"], bounds_error=False)(experiment_time)

    # Linear or nearest? TODO
    # print(stimulus_data.keys())
    if "stimulus_time" in visual_stimulus_data.keys():
        stimulus_extra_infos_data = interp1d(visual_stimulus_data["stimulus_time"], visual_stimulus_data["stimulus_data"], axis=0,bounds_error=False, kind='nearest')(experiment_time)
    else:
        stimulus_extra_infos_data = None

    # TODO HERE SHOULD NOT BE A BOUNDS ERROR, WORK ON THIS
    f_index_finder = interp1d(experiment_time, range(len(experiment_time)), kind='nearest')
    ind = np.where(visual_stimulus_data["stimulus_start_times"] >= experiment_time[0])

    stimulus_start_indices = f_index_finder(visual_stimulus_data["stimulus_start_times"][ind]).astype(np.int)
    stimulus_end_indices = f_index_finder(visual_stimulus_data["stimulus_end_times"][ind]).astype(np.int)
    stimulus_start_stimulus_numbers = visual_stimulus_data["stimulus_start_indices"][ind]
    stimulus_result_info = visual_stimulus_data["stimulus_result_info"][ind]



    head_right_eye_angle_event_start_indices, head_right_eye_angle_event_end_indices = find_events(head_right_eye_angle,
                                                                                                   dt=dt, window=0.05,
                                                                                                   start_threshold=1,
                                                                                                   end_threshold=0.1,
                                                                                                   minimal_inter_event_time=0.3)
    head_left_eye_angle_event_start_indices, head_left_eye_angle_event_end_indices = find_events(head_left_eye_angle,
                                                                                                 dt=dt, window=0.05,
                                                                                                 start_threshold=1,
                                                                                                 end_threshold=0.1,
                                                                                                 minimal_inter_event_time=0.3)
    tail_tip_deflection_event_start_indices, tail_tip_deflection_event_end_indices = find_events(tail_tip_deflection,
                                                                                                 dt=dt, window=0.05,
                                                                                                 start_threshold=1,
                                                                                                 end_threshold=0.1,
                                                                                                 minimal_inter_event_time=0.3)

    # refine the start indices for the tail tip deflection, find a local maximum, or minimum (important for aligning bouts)
    for i in range(len(tail_tip_deflection_event_start_indices)):
        window_diff = np.diff(tail_tip_deflection[int(tail_tip_deflection_event_start_indices[i] - 0.1 / dt):
                                                  tail_tip_deflection_event_start_indices[i]])

        ind = \
        np.where(((window_diff[:-1] > 0) & (window_diff[1:] < 0)) | ((window_diff[:-1] < 0) & (window_diff[1:] > 0)))[0]
        if len(ind) > 0:
            tail_tip_deflection_event_start_indices[i] = int(tail_tip_deflection_event_start_indices[i] - 0.1 / dt) + \
                                                         ind[-1] + 1

    # get the tail vigor
    tail_vigor = pd.rolling_var(tail_tip_deflection, window=int(0.05 / dt), center=True)

    ################
    # TODO: in the future, also align the 2P data

    all_behavior_data = dict({ "experiment_time": experiment_time,
                               "tail_shape_xs": tail_shape_xs,
                               "tail_shape_ys": tail_shape_ys,
                               "tail_tip_deflection": tail_tip_deflection,
                               "tail_vigor": tail_vigor,
                               "right_eye_angle": head_right_eye_angle,
                               "left_eye_angle": head_left_eye_angle,
                               "stimulus_extra_infos_data": stimulus_extra_infos_data,
                               "right_eye_angle_event_start_indices": head_right_eye_angle_event_start_indices,
                               "right_eye_angle_event_end_indices": head_right_eye_angle_event_end_indices,
                               "left_eye_angle_event_start_indices": head_left_eye_angle_event_start_indices,
                               "left_eye_angle_event_end_indices": head_left_eye_angle_event_end_indices,
                               "tail_tip_deflection_event_start_indices": tail_tip_deflection_event_start_indices,
                               "tail_tip_deflection_event_end_indices": tail_tip_deflection_event_end_indices,
                               "stimulus_start_indices": stimulus_start_indices,
                               "stimulus_end_indices": stimulus_end_indices,
                               "stimulus_result_info": stimulus_result_info,
                               "stimulus_start_stimulus_numbers": stimulus_start_stimulus_numbers
                               })

    return all_behavior_data

def get_aligned_data_embedded_4fish_setup(path, trial, dt, interpolate_tail_shape=False):

    filename_stimulus_data = "trial{:03d}_stimulus_data.npz".format(trial)
    filename_head_tail_data = "trial{:03d}_head_tail_data.npz".format(trial)

    stimulus_data = np.load(os.path.join(path, filename_stimulus_data))
    head_tail_data = np.load(os.path.join(path, filename_head_tail_data))

    camera_time = head_tail_data["camera_time"]

    experiment_time = np.arange(camera_time[0], camera_time[-1], dt)

    if interpolate_tail_shape == True:
        tail_shape_xs = interp1d(camera_time, head_tail_data["tail_shape_xs"], axis=0, bounds_error=False)(experiment_time)
        tail_shape_ys = interp1d(camera_time, head_tail_data["tail_shape_ys"], axis=0, bounds_error=False)(experiment_time)
    else:
        tail_shape_xs = None
        tail_shape_ys = None

    tail_tip_deflection = interp1d(camera_time, head_tail_data["tail_tip_deflection"], bounds_error=False)(experiment_time)

    left_eye_angle = interp1d(camera_time, head_tail_data["left_eye_angle"], bounds_error=False)(experiment_time)
    right_eye_angle = interp1d(camera_time, head_tail_data["right_eye_angle"], bounds_error=False)(experiment_time)

    # Linear or nearest? TODO
    if "stimulus_time" in stimulus_data.keys():
        stimulus_extra_infos_data = interp1d(stimulus_data["stimulus_time"], stimulus_data["stimulus_data"], axis=0, bounds_error=False)(experiment_time)
    else:
        stimulus_extra_infos_data = None

    # TODO HERE SHOULD NOT BE A BOUNDS ERROR, WORK ON THIS
    f_index_finder = interp1d(experiment_time, range(len(experiment_time)), kind='nearest', bounds_error=False)

    stimulus_start_indices = f_index_finder(stimulus_data["stimulus_start_times"]).astype(np.int)
    stimulus_start_stimulus_numbers = stimulus_data["stimulus_start_indices"]

    right_eye_angle_event_start_indices, right_eye_angle_event_end_indices = find_events(right_eye_angle, dt = dt, window = 0.05, start_threshold = 10, end_threshold = 2, minimal_inter_event_time = 0.3)
    left_eye_angle_event_start_indices, left_eye_angle_event_end_indices = find_events(left_eye_angle, dt = dt, window = 0.05, start_threshold = 10, end_threshold = 2, minimal_inter_event_time = 0.3)
    tail_tip_deflection_event_start_indices, tail_tip_deflection_event_end_indices = find_events(tail_tip_deflection, dt = dt, window = 0.05, start_threshold = 1.5, end_threshold = 0.5, minimal_inter_event_time = 0.3)

    # refine the start indices for the tail tip deflection, find a local maximum, or minimum (important for aligning bouts)
    for i in range(len(tail_tip_deflection_event_start_indices)):
        window_diff = np.diff(tail_tip_deflection[int(tail_tip_deflection_event_start_indices[i] - 0.1 / dt):tail_tip_deflection_event_start_indices[i]])

        ind = np.where(((window_diff[:-1] > 0) & (window_diff[1:] < 0)) | ((window_diff[:-1] < 0) & (window_diff[1:] > 0)))[0]
        if len(ind) > 0:
            tail_tip_deflection_event_start_indices[i] = int(tail_tip_deflection_event_start_indices[i] - 0.1 / dt) + ind[-1] + 1


    # get the tail vigor
    tail_vigor = pd.rolling_var(tail_tip_deflection, window=int(0.05 / dt), center=True)

    all_behavior_data = dict({ "experiment_time": experiment_time,
                               "tail_shape_xs": tail_shape_xs,
                               "tail_shape_ys": tail_shape_ys,
                               "tail_tip_deflection": tail_tip_deflection,
                               "tail_vigor": tail_vigor,
                               "right_eye_angle": right_eye_angle,
                               "left_eye_angle": left_eye_angle,
                               "stimulus_extra_infos_data": stimulus_extra_infos_data,
                               "right_eye_angle_event_start_indices": right_eye_angle_event_start_indices,
                               "right_eye_angle_event_end_indices": right_eye_angle_event_end_indices,
                               "left_eye_angle_event_start_indices": left_eye_angle_event_start_indices,
                               "left_eye_angle_event_end_indices": left_eye_angle_event_end_indices,
                               "tail_tip_deflection_event_start_indices": tail_tip_deflection_event_start_indices,
                               "tail_tip_deflection_event_end_indices": tail_tip_deflection_event_end_indices,
                               "stimulus_start_indices": stimulus_start_indices,
                               "stimulus_end_indices": np.r_[stimulus_start_indices[1:], len(experiment_time)],
                               "stimulus_start_stimulus_numbers": stimulus_start_stimulus_numbers
                               })

    return all_behavior_data


def get_aligned_data_embedded_4fish_setup2(path, trial, dt, interpolate_tail_shape=False):

    filename_stimulus_data = "trial{:03d}_stimulus_data.npz".format(trial)
    filename_head_tail_data = "trial{:03d}_head_tail_data.npz".format(trial)

    stimulus_data = np.load(os.path.join(path, filename_stimulus_data))
    head_tail_data = np.load(os.path.join(path, filename_head_tail_data))

    camera_time = head_tail_data["camera_time"]

    experiment_time = np.arange(camera_time[0], camera_time[-1], dt)

    if interpolate_tail_shape == True:
        tail_shape_xs = interp1d(camera_time, head_tail_data["tail_shape_xs"], axis=0, bounds_error=False)(experiment_time)
        tail_shape_ys = interp1d(camera_time, head_tail_data["tail_shape_ys"], axis=0, bounds_error=False)(experiment_time)
    else:
        tail_shape_xs = None
        tail_shape_ys = None

    tail_tip_deflection = interp1d(camera_time, head_tail_data["tail_tip_deflection"], bounds_error=False)(experiment_time)

    left_eye_angle = interp1d(camera_time, head_tail_data["left_eye_angle"], bounds_error=False)(experiment_time)
    right_eye_angle = interp1d(camera_time, head_tail_data["right_eye_angle"], bounds_error=False)(experiment_time)

    # Linear or nearest? TODO
    #print(stimulus_data.keys())
    if "stimulus_time" in stimulus_data.keys():
        stimulus_extra_infos_data = interp1d(stimulus_data["stimulus_time"], stimulus_data["stimulus_data"], axis=0, bounds_error=False, kind='nearest')(experiment_time)
    else:
        stimulus_extra_infos_data = None

    # TODO HERE SHOULD NOT BE A BOUNDS ERROR, WORK ON THIS
    f_index_finder = interp1d(experiment_time, range(len(experiment_time)), kind='nearest')
    ind = np.where(stimulus_data["stimulus_start_times"] >= experiment_time[0])

    stimulus_start_indices = f_index_finder(stimulus_data["stimulus_start_times"][ind]).astype(np.int)
    stimulus_end_indices = f_index_finder(stimulus_data["stimulus_end_times"][ind]).astype(np.int)
    stimulus_start_stimulus_numbers = stimulus_data["stimulus_start_indices"][ind]
    stimulus_result_info = stimulus_data["stimulus_result_info"][ind]

    right_eye_angle_event_start_indices, right_eye_angle_event_end_indices = find_events(right_eye_angle, dt = dt, window = 0.05, start_threshold = 5, end_threshold = 1, minimal_inter_event_time = 0.3)
    left_eye_angle_event_start_indices, left_eye_angle_event_end_indices = find_events(left_eye_angle, dt = dt, window = 0.05, start_threshold = 5, end_threshold = 1, minimal_inter_event_time = 0.3)
    tail_tip_deflection_event_start_indices, tail_tip_deflection_event_end_indices = find_events(tail_tip_deflection, dt = dt, window = 0.05, start_threshold = 1.5, end_threshold = 0.5, minimal_inter_event_time = 0.3)

    # refine the start indices for the tail tip deflection, find a local maximum, or minimum (important for aligning bouts)
    #for i in range(len(tail_tip_deflection_event_start_indices)):
    #    window_diff = np.diff(tail_tip_deflection[int(tail_tip_deflection_event_start_indices[i] - 0.1 / dt):tail_tip_deflection_event_start_indices[i]])

    #    ind = np.where(((window_diff[:-1] > 0) & (window_diff[1:] < 0)) | ((window_diff[:-1] < 0) & (window_diff[1:] > 0)))[0]
    #    if len(ind) > 0:
    #        tail_tip_deflection_event_start_indices[i] = int(tail_tip_deflection_event_start_indices[i] - 0.1 / dt) + ind[-1] + 1


    # get the tail vigor
    tail_vigor = pd.rolling_var(tail_tip_deflection, window=int(0.05 / dt), center=True)

    all_behavior_data = dict({ "experiment_time": experiment_time,
                               "tail_shape_xs": tail_shape_xs,
                               "tail_shape_ys": tail_shape_ys,
                               "tail_tip_deflection": tail_tip_deflection,
                               "tail_vigor": tail_vigor,
                               "right_eye_angle": right_eye_angle,
                               "left_eye_angle": left_eye_angle,
                               "stimulus_extra_infos_data": stimulus_extra_infos_data,
                               "right_eye_angle_event_start_indices": right_eye_angle_event_start_indices,
                               "right_eye_angle_event_end_indices": right_eye_angle_event_end_indices,
                               "left_eye_angle_event_start_indices": left_eye_angle_event_start_indices,
                               "left_eye_angle_event_end_indices": left_eye_angle_event_end_indices,
                               "tail_tip_deflection_event_start_indices": tail_tip_deflection_event_start_indices,
                               "tail_tip_deflection_event_end_indices": tail_tip_deflection_event_end_indices,
                               "stimulus_start_indices": stimulus_start_indices,
                               "stimulus_end_indices": stimulus_end_indices,
                               "stimulus_result_info": stimulus_result_info,
                               "stimulus_start_stimulus_numbers": stimulus_start_stimulus_numbers
                               })

    return all_behavior_data
