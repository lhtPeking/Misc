# -*- coding: utf-8 -*-

import sys
sys._excepthook = sys.excepthook
def exception_hook(exctype, value, traceback):
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)
sys.excepthook = exception_hook

if __name__ == "__main__":

    from shared import Shared

    shared = Shared()
    shared.start_threads()

    from PyQt5 import QtCore, QtGui, uic, QtWidgets
    import os
    import pyqtgraph as pg
    import numpy as np
    import pickle

    class Embedded_4fish_gui(QtWidgets.QDialog):
        def __init__(self, parent=None):
            QtWidgets.QWidget.__init__(self, parent)

            self.shared = shared
            path = os.path.dirname(__file__)
            os.chdir(path)

            pg.setConfigOption('background', pg.mkColor(20 / 255.))
            pg.setConfigOption('foreground', 'w')

            uic.loadUi(os.path.join(path, "embedded_4fish_gui.ui"), self)
  
            # scale all the widgets (4k, windows scaling problem)
            fontscale = 1  # 2.2
            sizescale = 1 #2.2

            # scale the application window
            size = self.size()
            self.resize(size.width() * sizescale, size.height() * sizescale)

            for widget in self.findChildren(QtWidgets.QWidget):
                size = widget.size()

                widget.resize(size.width() * sizescale, size.height() * sizescale)
                widget.move(widget.x() * sizescale, widget.y() * sizescale)

                font = widget.font()

                font.setPointSize(font.pointSize() * fontscale)
                widget.setFont(font)

            ########################
            ##### Fish Selection
            self.comboBox_select_fish.addItem("Fish 0 (upper left)")
            self.comboBox_select_fish.addItem("Fish 1 (upper right)")
            self.comboBox_select_fish.addItem("Fish 2 (lower left)")
            self.comboBox_select_fish.addItem("Fish 3 (lower right)")
            self.comboBox_select_fish.setCurrentIndex(self.shared.fish_index_display.value)

            self.comboBox_select_fish.activated.connect(self.comboBox_select_fish_activated)

            ########################
            ##### Fish Configuration
            self.checkBox_fish_configuration_use_fish.clicked.connect(self.checkBox_fish_configuration_use_fish_clicked)
            self.spinBox_fish_configuration_ID.valueChanged.connect(self.spinBox_fish_configuration_ID_valueChanged)
            self.lineEdit_fish_configuration_genotype.textChanged.connect(self.lineEdit_fish_configuration_genotype_textChanged)
            self.lineEdit_fish_configuration_age.textChanged.connect(self.lineEdit_fish_configuration_age_textChanged)
            self.lineEdit_fish_configuration_comment.textChanged.connect(self.lineEdit_fish_configuration_comment_textChanged)

            ########################
            ##### Stimulus Configuration
            self.lineEdit_stimulus_configuration_stimulus_path.textChanged.connect(self.lineEdit_stimulus_configuration_stimulus_path_textChanged)
            self.pushButton_stimulus_configuration_load_stimulus_path.clicked.connect(self.pushButton_stimulus_configuration_load_stimulus_path_clicked)

            self.doubleSpinBox_stimulus_configuration_set_x_position.valueChanged.connect(self.doubleSpinBox_stimulus_configuration_set_x_position_valueChanged)
            self.doubleSpinBox_stimulus_configuration_set_y_position.valueChanged.connect(self.doubleSpinBox_stimulus_configuration_set_y_position_valueChanged)
            self.doubleSpinBox_stimulus_configuration_set_scale.valueChanged.connect(self.doubleSpinBox_stimulus_configuration_set_scale_valueChanged)
            self.doubleSpinBox_stimulus_configuration_set_rotation.valueChanged.connect(self.doubleSpinBox_stimulus_configuration_set_rotation_valueChanged)
            self.pushButton_stimulus_configuration_start_test_stimulus_index.clicked.connect(self.pushButton_stimulus_configuration_start_test_stimulus_index_clicked)

            #######################
            #### Camera Configuration
            self.doubleSpinBox_fish_camera_set_gain.valueChanged.connect(self.doubleSpinBox_fish_camera_set_gain_valueChanged)
            self.doubleSpinBox_fish_camera_set_shutter.valueChanged.connect(self.doubleSpinBox_fish_camera_set_shutter_valueChanged)
            self.spinBox_fish_camera_set_roi_x.valueChanged.connect(self.spinBox_fish_camera_set_roi_x_valueChanged)
            self.spinBox_fish_camera_set_roi_y.valueChanged.connect(self.spinBox_fish_camera_set_roi_y_valueChanged)
            self.spinBox_fish_camera_set_roi_width.valueChanged.connect(self.spinBox_fish_camera_set_roi_width_valueChanged)
            self.spinBox_fish_camera_set_roi_height.valueChanged.connect(self.spinBox_fish_camera_set_roi_height_valueChanged)

            #######################
            ### Eye Tracking Configuration
            self.spinBox_eye_tracking_configuration_left_eye_x.valueChanged.connect(self.spinBox_eye_tracking_configuration_left_eye_x_valueChanged)
            self.spinBox_eye_tracking_configuration_left_eye_y.valueChanged.connect(self.spinBox_eye_tracking_configuration_left_eye_y_valueChanged)
            self.spinBox_eye_tracking_configuration_right_eye_x.valueChanged.connect(self.spinBox_eye_tracking_configuration_right_eye_x_valueChanged)
            self.spinBox_eye_tracking_configuration_right_eye_y.valueChanged.connect(self.spinBox_eye_tracking_configuration_right_eye_y_valueChanged)
            self.spinBox_eye_tracking_configuration_threshold.valueChanged.connect(self.spinBox_eye_tracking_configuration_threshold_valueChanged)
            self.spinBox_eye_tracking_configuration_radius.valueChanged.connect(self.spinBox_eye_tracking_configuration_radius_valueChanged)
            self.spinBox_eye_tracking_configuration_angles.valueChanged.connect(self.spinBox_eye_tracking_configuration_angles_valueChanged)
            self.checkBox_eye_tracking_configuration_display_tracking_process.clicked.connect(self.checkBox_eye_tracking_configuration_display_tracking_process_clicked)

            #######################
            ### Tail Tracking Configuration
            self.spinBox_tail_tracking_set_x0.valueChanged.connect(self.spinBox_tail_tracking_set_x0_valueChanged)
            self.spinBox_tail_tracking_set_y0.valueChanged.connect(self.spinBox_tail_tracking_set_y0_valueChanged)
            self.spinBox_tail_tracking_set_nodes.valueChanged.connect(self.spinBox_tail_tracking_set_nodes_valueChanged)
            self.doubleSpinBox_tail_tracking_set_bout_start_vigor.valueChanged.connect(self.doubleSpinBox_tail_tracking_set_bout_start_vigor_valueChanged)
            self.doubleSpinBox_tail_tracking_set_bout_end_vigor.valueChanged.connect(self.doubleSpinBox_tail_tracking_set_bout_end_vigor_valueChanged)

            #######################
            ### Experiment Configuration
            self.lineEdit_experiment_configuration_storage_root_path.textChanged.connect(self.lineEdit_experiment_configuration_storage_root_path_textChanged)
            self.pushButton_experiment_configuration_load_storage_root_path.clicked.connect(self.pushButton_experiment_configuration_load_storage_root_path_clicked)
            self.spinBox_experiment_configuration_number_of_trials.valueChanged.connect(self.spinBox_experiment_configuration_number_of_trials_valueChanged)
            self.spinBox_experiment_configuration_trial_time.valueChanged.connect(self.spinBox_experiment_configuration_trial_time_valueChanged)

            self.pushButton_start_stop_experiment.clicked.connect(self.pushButton_start_stop_experiment_clicked)
            self.checkBox_experiment_configuration_store_head_tail_data.clicked.connect(self.checkBox_experiment_configuration_store_head_tail_data_clicked)
            self.checkBox_experiment_configuration_store_head_tail_movie.clicked.connect(self.checkBox_experiment_configuration_store_head_tail_movie_clicked)

            ########################
            ##### Fish display
            self.pyqtgraph_fish_camera_display.hideAxis('left')
            self.pyqtgraph_fish_camera_display.hideAxis('bottom')
            self.pyqtgraph_fish_camera_display.setAspectLocked()
            self.pyqtgraph_fish_camera_display.setXRange(-1, 1)

            self.pyqtgraph_fish_camera_display_image_item = pg.ImageItem(image=np.zeros((1, 1)))
            self.pyqtgraph_fish_camera_display_tail_shape_item = pg.PlotDataItem()

            self.pyqtgraph_fish_camera_display.addItem(self.pyqtgraph_fish_camera_display_image_item) 
            self.pyqtgraph_fish_camera_display.addItem(self.pyqtgraph_fish_camera_display_tail_shape_item)

            self.load_program_configuration()

            # automatically suggest some fish IDs
            for fish_index in range(4):
                self.shared.fish_configuration_ID[fish_index].value = fish_index

            self.update_edit_fields()

            # update all cameras to the rois (roi update also updates gain and shutter)
            for fish_index in range(4):
                self.shared.fish_camera_update_roi_requested[fish_index].value = 1


            self.update_gui_timer = QtCore.QTimer()
            self.update_gui_timer.timeout.connect(self.update_gui)
            self.update_gui_timer.start(30)

        def load_program_configuration(self):

            try:

                ### if the configuration info can't be loaded, use the other setup's pickle file
                ### after you loaded this setup1.pickle file and run embedded_4fish.py, setup0.pickle file will be updated, so you can use setup0.pickle file as original.
                # [fish_dict, data_dict] = pickle.load(
                #     open("program_configuration_setup1.pickle", "rb"))
                # print("program_configuration_setup1.pickle")

                [fish_dict, data_dict] = pickle.load(open("program_configuration_setup{}.pickle".format(self.shared.setup_ID), "rb"))
                print("program_configuration_setup{}.pickle".format(self.shared.setup_ID))


                for fish_index in range(4):
                    self.shared.fish_configuration_use_fish[fish_index].value = fish_dict[fish_index]["fish_configuration_use_fish"]

                    self.shared.stimulus_configuration_set_x_position[fish_index].value = fish_dict[fish_index]["stimulus_configuration_set_x_position"]
                    self.shared.stimulus_configuration_set_y_position[fish_index].value = fish_dict[fish_index]["stimulus_configuration_set_y_position"]
                    self.shared.stimulus_configuration_set_scale[fish_index].value = fish_dict[fish_index]["stimulus_configuration_set_scale"]
                    self.shared.stimulus_configuration_set_rotation[fish_index].value = fish_dict[fish_index]["stimulus_configuration_set_rotation"]

                    self.shared.fish_camera_set_gain[fish_index].value = fish_dict[fish_index]["fish_camera_set_gain"]
                    self.shared.fish_camera_set_shutter[fish_index].value = fish_dict[fish_index]["fish_camera_set_shutter"]
                    self.shared.fish_camera_set_roi_x[fish_index].value = fish_dict[fish_index]["fish_camera_set_roi_x"]
                    self.shared.fish_camera_set_roi_y[fish_index].value = fish_dict[fish_index]["fish_camera_set_roi_y"]
                    self.shared.fish_camera_set_roi_width[fish_index].value = fish_dict[fish_index]["fish_camera_set_roi_width"]
                    self.shared.fish_camera_set_roi_height[fish_index].value = fish_dict[fish_index]["fish_camera_set_roi_height"]

                    self.shared.eye_tracking_configuration_left_eye_x[fish_index].value = fish_dict[fish_index]["eye_tracking_configuration_left_eye_x"]
                    self.shared.eye_tracking_configuration_left_eye_y[fish_index].value = fish_dict[fish_index]["eye_tracking_configuration_left_eye_y"]
                    self.shared.eye_tracking_configuration_right_eye_x[fish_index].value = fish_dict[fish_index]["eye_tracking_configuration_right_eye_x"]
                    self.shared.eye_tracking_configuration_right_eye_y[fish_index].value = fish_dict[fish_index]["eye_tracking_configuration_right_eye_y"]
                    self.shared.eye_tracking_configuration_threshold[fish_index].value = fish_dict[fish_index]["eye_tracking_configuration_threshold"]
                    self.shared.eye_tracking_configuration_radius[fish_index].value = fish_dict[fish_index]["eye_tracking_configuration_radius"]
                    self.shared.eye_tracking_configuration_angles[fish_index].value = fish_dict[fish_index]["eye_tracking_configuration_angles"]
                    self.shared.eye_tracking_configuration_display_tracking_process[fish_index].value = fish_dict[fish_index]["eye_tracking_configuration_display_tracking_process"]

                    self.shared.tail_tracking_set_x0[fish_index].value = fish_dict[fish_index]["tail_tracking_set_x0"]
                    self.shared.tail_tracking_set_y0[fish_index].value = fish_dict[fish_index]["tail_tracking_set_y0"]
                    self.shared.tail_tracking_set_nodes[fish_index].value = fish_dict[fish_index]["tail_tracking_set_nodes"]
                    self.shared.tail_tracking_set_bout_start_vigor[fish_index].value = fish_dict[fish_index]["tail_tracking_set_bout_start_vigor"]
                    self.shared.tail_tracking_set_bout_end_vigor[fish_index].value = fish_dict[fish_index]["tail_tracking_set_bout_end_vigor"]

                    #self.shared.recorded_tail_tracking_xs[fish_index] = np.zeros(fish_dict[fish_index]["tail_tracking_set_nodes"])
                    #self.shared.recorded_tail_tracking_ys[fish_index] = np.zeros(fish_dict[fish_index]["tail_tracking_set_nodes"])
                    #self.shared.recorded_timeindex = np.arange(1000)


                self.shared.experiment_configuration_number_of_trials.value = data_dict["experiment_configuration_number_of_trials"]
                self.shared.experiment_configuration_trial_time.value = data_dict["experiment_configuration_trial_time"]

                #filepath = data_dict["experiment_configuration_storage_root_path"].encode()
                #self.shared.experiment_configuration_storage_root_path[:len(filepath)] = filepath
                #self.shared.experiment_configuration_storage_root_path_l.value = len(filepath)

                self.shared.experiment_configuration_store_head_tail_data.value = data_dict["experiment_configuration_store_head_tail_data"]
                self.shared.experiment_configuration_store_head_tail_movie.value = data_dict["experiment_configuration_store_head_tail_movie"]

            except Exception as e:
                print(e)


        def save_program_configuration(self):

            fish_dict = [dict({}) for _ in range(4)]

            for fish_index in range(4):

                fish_dict[fish_index]["fish_configuration_use_fish"] = self.shared.fish_configuration_use_fish[fish_index].value

                fish_dict[fish_index]["stimulus_configuration_set_x_position"] = self.shared.stimulus_configuration_set_x_position[fish_index].value
                fish_dict[fish_index]["stimulus_configuration_set_y_position"] = self.shared.stimulus_configuration_set_y_position[fish_index].value
                fish_dict[fish_index]["stimulus_configuration_set_scale"] = self.shared.stimulus_configuration_set_scale[fish_index].value
                fish_dict[fish_index]["stimulus_configuration_set_rotation"] = self.shared.stimulus_configuration_set_rotation[fish_index].value

                fish_dict[fish_index]["fish_camera_set_gain"] = self.shared.fish_camera_set_gain[fish_index].value
                fish_dict[fish_index]["fish_camera_set_shutter"] = self.shared.fish_camera_set_shutter[fish_index].value
                fish_dict[fish_index]["fish_camera_set_roi_x"] = self.shared.fish_camera_set_roi_x[fish_index].value
                fish_dict[fish_index]["fish_camera_set_roi_y"] = self.shared.fish_camera_set_roi_y[fish_index].value
                fish_dict[fish_index]["fish_camera_set_roi_width"] = self.shared.fish_camera_set_roi_width[fish_index].value
                fish_dict[fish_index]["fish_camera_set_roi_height"] = self.shared.fish_camera_set_roi_height[fish_index].value

                fish_dict[fish_index]["eye_tracking_configuration_left_eye_x"] = self.shared.eye_tracking_configuration_left_eye_x[fish_index].value
                fish_dict[fish_index]["eye_tracking_configuration_left_eye_y"] = self.shared.eye_tracking_configuration_left_eye_y[fish_index].value
                fish_dict[fish_index]["eye_tracking_configuration_right_eye_x"] = self.shared.eye_tracking_configuration_right_eye_x[fish_index].value
                fish_dict[fish_index]["eye_tracking_configuration_right_eye_y"] = self.shared.eye_tracking_configuration_right_eye_y[fish_index].value
                fish_dict[fish_index]["eye_tracking_configuration_threshold"] = self.shared.eye_tracking_configuration_threshold[fish_index].value
                fish_dict[fish_index]["eye_tracking_configuration_radius"] = self.shared.eye_tracking_configuration_radius[fish_index].value
                fish_dict[fish_index]["eye_tracking_configuration_angles"] = self.shared.eye_tracking_configuration_angles[fish_index].value
                fish_dict[fish_index]["eye_tracking_configuration_display_tracking_process"] = self.shared.eye_tracking_configuration_display_tracking_process[fish_index].value

                fish_dict[fish_index]["tail_tracking_set_x0"] = self.shared.tail_tracking_set_x0[fish_index].value
                fish_dict[fish_index]["tail_tracking_set_y0"] = self.shared.tail_tracking_set_y0[fish_index].value
                fish_dict[fish_index]["tail_tracking_set_nodes"] = self.shared.tail_tracking_set_nodes[fish_index].value
                fish_dict[fish_index]["tail_tracking_set_bout_start_vigor"] = self.shared.tail_tracking_set_bout_start_vigor[fish_index].value
                fish_dict[fish_index]["tail_tracking_set_bout_end_vigor"] = self.shared.tail_tracking_set_bout_end_vigor[fish_index].value


            # fish unspecific variables
            data_dict = dict({})
            data_dict["experiment_configuration_number_of_trials"] = self.shared.experiment_configuration_number_of_trials.value
            data_dict["experiment_configuration_trial_time"] = self.shared.experiment_configuration_trial_time.value
            data_dict["experiment_configuration_storage_root_path"] = bytearray(self.shared.experiment_configuration_storage_root_path[:self.shared.experiment_configuration_storage_root_path_l.value]).decode()
            data_dict["experiment_configuration_store_head_tail_data"] = self.shared.experiment_configuration_store_head_tail_data.value
            data_dict["experiment_configuration_store_head_tail_movie"] = self.shared.experiment_configuration_store_head_tail_movie.value

            try:
                pickle.dump([fish_dict, data_dict] , open("program_configuration_setup{}.pickle".format(self.shared.setup_ID), "wb"))
            except Exception as e:
                print(e)


        def update_edit_fields(self):
            fish_index = self.shared.fish_index_display.value

            # signals need to be blocked when setting manually, otherwise cameras reset, etc.
            for widget in self.findChildren(QtWidgets.QWidget):
                widget.blockSignals(True)

            self.checkBox_fish_configuration_use_fish.setCheckState(QtCore.Qt.Checked if self.shared.fish_configuration_use_fish[fish_index].value == 1 else QtCore.Qt.Unchecked)

            self.spinBox_fish_configuration_ID.setValue(self.shared.fish_configuration_ID[fish_index].value)
            self.lineEdit_fish_configuration_genotype.setText(bytearray(self.shared.fish_configuration_genotype[fish_index][:self.shared.fish_configuration_genotype_l[fish_index].value]).decode())
            self.lineEdit_fish_configuration_age.setText(bytearray(self.shared.fish_configuration_age[fish_index][:self.shared.fish_configuration_age_l[fish_index].value]).decode())
            self.lineEdit_fish_configuration_comment.setText(bytearray(self.shared.fish_configuration_comment[fish_index][:self.shared.fish_configuration_comment_l[fish_index].value]).decode())

            self.doubleSpinBox_stimulus_configuration_set_x_position.setValue(self.shared.stimulus_configuration_set_x_position[fish_index].value)
            self.doubleSpinBox_stimulus_configuration_set_y_position.setValue(self.shared.stimulus_configuration_set_y_position[fish_index].value)
            self.doubleSpinBox_stimulus_configuration_set_scale.setValue(self.shared.stimulus_configuration_set_scale[fish_index].value)
            self.doubleSpinBox_stimulus_configuration_set_rotation.setValue(self.shared.stimulus_configuration_set_rotation[fish_index].value)

            self.spinBox_eye_tracking_configuration_left_eye_x.setValue(self.shared.eye_tracking_configuration_left_eye_x[fish_index].value)
            self.spinBox_eye_tracking_configuration_left_eye_y.setValue(self.shared.eye_tracking_configuration_left_eye_y[fish_index].value)
            self.spinBox_eye_tracking_configuration_right_eye_x.setValue(self.shared.eye_tracking_configuration_right_eye_x[fish_index].value)
            self.spinBox_eye_tracking_configuration_right_eye_y.setValue(self.shared.eye_tracking_configuration_right_eye_y[fish_index].value)
            self.spinBox_eye_tracking_configuration_threshold.setValue(self.shared.eye_tracking_configuration_threshold[fish_index].value)
            self.spinBox_eye_tracking_configuration_radius.setValue(self.shared.eye_tracking_configuration_radius[fish_index].value)
            self.spinBox_eye_tracking_configuration_angles.setValue(self.shared.eye_tracking_configuration_angles[fish_index].value)

            self.checkBox_eye_tracking_configuration_display_tracking_process.setCheckState(QtCore.Qt.Checked if self.shared.eye_tracking_configuration_display_tracking_process[fish_index].value == 1 else QtCore.Qt.Unchecked)

            self.spinBox_tail_tracking_set_x0.setValue(self.shared.tail_tracking_set_x0[fish_index].value)
            self.spinBox_tail_tracking_set_y0.setValue(self.shared.tail_tracking_set_y0[fish_index].value)
            self.spinBox_tail_tracking_set_nodes.setValue(self.shared.tail_tracking_set_nodes[fish_index].value)
            self.doubleSpinBox_tail_tracking_set_bout_start_vigor.setValue(self.shared.tail_tracking_set_bout_start_vigor[fish_index].value)
            self.doubleSpinBox_tail_tracking_set_bout_end_vigor.setValue(self.shared.tail_tracking_set_bout_end_vigor[fish_index].value)

            self.doubleSpinBox_fish_camera_set_gain.setValue(self.shared.fish_camera_set_gain[fish_index].value)
            self.doubleSpinBox_fish_camera_set_shutter.setValue(self.shared.fish_camera_set_shutter[fish_index].value)
            self.spinBox_fish_camera_set_roi_x.setValue(self.shared.fish_camera_set_roi_x[fish_index].value)
            self.spinBox_fish_camera_set_roi_y.setValue(self.shared.fish_camera_set_roi_y[fish_index].value)
            self.spinBox_fish_camera_set_roi_width.setValue(self.shared.fish_camera_set_roi_width[fish_index].value)
            self.spinBox_fish_camera_set_roi_height.setValue(self.shared.fish_camera_set_roi_height[fish_index].value)

            self.lineEdit_experiment_configuration_storage_root_path.setText(bytearray(self.shared.experiment_configuration_storage_root_path[:self.shared.experiment_configuration_storage_root_path_l.value]).decode())
            self.spinBox_experiment_configuration_number_of_trials.setValue(self.shared.experiment_configuration_number_of_trials.value)
            self.spinBox_experiment_configuration_trial_time.setValue(self.shared.experiment_configuration_trial_time.value)
            self.checkBox_experiment_configuration_store_head_tail_data.setCheckState(QtCore.Qt.Checked if self.shared.experiment_configuration_store_head_tail_data.value == 1 else QtCore.Qt.Unchecked)
            self.checkBox_experiment_configuration_store_head_tail_movie.setCheckState(QtCore.Qt.Checked if self.shared.experiment_configuration_store_head_tail_movie.value == 1 else QtCore.Qt.Unchecked)

            # and allow the widgets to send messages again
            for widget in self.findChildren(QtWidgets.QWidget):
                widget.blockSignals(False)

        def comboBox_select_fish_activated(self):
            self.shared.fish_index_display.value = self.comboBox_select_fish.currentIndex()

            self.update_edit_fields()

        def checkBox_fish_configuration_use_fish_clicked(self):
            if self.checkBox_fish_configuration_use_fish.checkState() == QtCore.Qt.Checked:
                self.shared.fish_configuration_use_fish[self.shared.fish_index_display.value].value = 1
            else:
                self.shared.fish_configuration_use_fish[self.shared.fish_index_display.value].value = 0

        def spinBox_fish_configuration_ID_valueChanged(self):
            self.shared.fish_configuration_ID[self.shared.fish_index_display.value].value = self.spinBox_fish_configuration_ID.value()

        def lineEdit_fish_configuration_genotype_textChanged(self):
            text = self.lineEdit_fish_configuration_genotype.text().encode()

            self.shared.fish_configuration_genotype[self.shared.fish_index_display.value][:len(text)] = text
            self.shared.fish_configuration_genotype_l[self.shared.fish_index_display.value].value = len(text)

        def lineEdit_fish_configuration_age_textChanged(self):

            text = self.lineEdit_fish_configuration_age.text().encode()

            self.shared.fish_configuration_age[self.shared.fish_index_display.value][:len(text)] = text
            self.shared.fish_configuration_age_l[self.shared.fish_index_display.value].value = len(text)

        def lineEdit_fish_configuration_comment_textChanged(self):

            text = self.lineEdit_fish_configuration_comment.text().encode()

            self.shared.fish_configuration_comment[self.shared.fish_index_display.value][:len(text)] = text
            self.shared.fish_configuration_comment_l[self.shared.fish_index_display.value].value = len(text)

        def lineEdit_stimulus_configuration_stimulus_path_textChanged(self):
            full_path_to_module = self.lineEdit_stimulus_configuration_stimulus_path.text().encode()

            self.shared.stimulus_configuration_stimulus_path[:len(full_path_to_module)] = full_path_to_module
            self.shared.stimulus_configuration_stimulus_path_l.value = len(full_path_to_module)
            self.shared.stimulus_configuration_stimulus_path_update_requested.value = 1

        def pushButton_stimulus_configuration_load_stimulus_path_clicked(self):
            try:
                full_path_to_module = os.path.abspath(QtWidgets.QFileDialog.getOpenFileName()[0])
                self.lineEdit_stimulus_configuration_stimulus_path.setText(full_path_to_module)
            except:
                print("Weird...")

        def doubleSpinBox_stimulus_configuration_set_x_position_valueChanged(self):
            self.shared.stimulus_configuration_set_x_position[self.shared.fish_index_display.value].value = self.doubleSpinBox_stimulus_configuration_set_x_position.value()

        def doubleSpinBox_stimulus_configuration_set_y_position_valueChanged(self):
            self.shared.stimulus_configuration_set_y_position[self.shared.fish_index_display.value].value = self.doubleSpinBox_stimulus_configuration_set_y_position.value()

        def doubleSpinBox_stimulus_configuration_set_scale_valueChanged(self):
            self.shared.stimulus_configuration_set_scale[self.shared.fish_index_display.value].value = self.doubleSpinBox_stimulus_configuration_set_scale.value()

        def doubleSpinBox_stimulus_configuration_set_rotation_valueChanged(self):
            self.shared.stimulus_configuration_set_rotation[self.shared.fish_index_display.value].value = self.doubleSpinBox_stimulus_configuration_set_rotation.value()

        def pushButton_stimulus_configuration_start_test_stimulus_index_clicked(self):
            if self.shared.stimulus_flow_control_currently_running[self.shared.fish_index_display.value].value == 0:
                self.shared.stimulus_flow_control_index[self.shared.fish_index_display.value].value = self.spinBox_stimulus_configuration_set_test_stimulus_index.value()
                self.shared.stimulus_flow_control_start_requested[self.shared.fish_index_display.value].value = 1
            else:
                self.shared.stimulus_flow_control_currently_running[self.shared.fish_index_display.value].value = 0

        def doubleSpinBox_fish_camera_set_gain_valueChanged(self):
            self.shared.fish_camera_set_gain[self.shared.fish_index_display.value].value = self.doubleSpinBox_fish_camera_set_gain.value()
            self.shared.fish_camera_update_gain_shutter_requested[self.shared.fish_index_display.value].value = 1

        def doubleSpinBox_fish_camera_set_shutter_valueChanged(self):
            self.shared.fish_camera_set_shutter[self.shared.fish_index_display.value].value = self.doubleSpinBox_fish_camera_set_shutter.value()
            self.shared.fish_camera_update_gain_shutter_requested[self.shared.fish_index_display.value].value = 1

        def spinBox_fish_camera_set_roi_x_valueChanged(self):
            self.shared.fish_camera_set_roi_x[self.shared.fish_index_display.value].value = self.spinBox_fish_camera_set_roi_x.value()
            self.shared.fish_camera_update_roi_requested[self.shared.fish_index_display.value].value = 1

        def spinBox_fish_camera_set_roi_y_valueChanged(self):
            self.shared.fish_camera_set_roi_y[self.shared.fish_index_display.value].value = self.spinBox_fish_camera_set_roi_y.value()
            self.shared.fish_camera_update_roi_requested[self.shared.fish_index_display.value].value = 1

        def spinBox_fish_camera_set_roi_width_valueChanged(self):
            self.shared.fish_camera_set_roi_width[self.shared.fish_index_display.value].value = self.spinBox_fish_camera_set_roi_width.value()
            self.shared.fish_camera_update_roi_requested[self.shared.fish_index_display.value].value = 1

        def spinBox_fish_camera_set_roi_height_valueChanged(self):
            self.shared.fish_camera_set_roi_height[self.shared.fish_index_display.value].value = self.spinBox_fish_camera_set_roi_height.value()
            self.shared.fish_camera_update_roi_requested[self.shared.fish_index_display.value].value = 1

        def spinBox_eye_tracking_configuration_left_eye_x_valueChanged(self):
            self.shared.eye_tracking_configuration_left_eye_x[self.shared.fish_index_display.value].value = self.spinBox_eye_tracking_configuration_left_eye_x.value()

        def spinBox_eye_tracking_configuration_left_eye_y_valueChanged(self):
            self.shared.eye_tracking_configuration_left_eye_y[self.shared.fish_index_display.value].value = self.spinBox_eye_tracking_configuration_left_eye_y.value()

        def spinBox_eye_tracking_configuration_right_eye_x_valueChanged(self):
            self.shared.eye_tracking_configuration_right_eye_x[self.shared.fish_index_display.value].value = self.spinBox_eye_tracking_configuration_right_eye_x.value()

        def spinBox_eye_tracking_configuration_right_eye_y_valueChanged(self):
            self.shared.eye_tracking_configuration_right_eye_y[self.shared.fish_index_display.value].value = self.spinBox_eye_tracking_configuration_right_eye_y.value()

        def spinBox_eye_tracking_configuration_threshold_valueChanged(self):
            self.shared.eye_tracking_configuration_threshold[self.shared.fish_index_display.value].value = self.spinBox_eye_tracking_configuration_threshold.value()

        def spinBox_eye_tracking_configuration_radius_valueChanged(self):
            self.shared.eye_tracking_configuration_radius[self.shared.fish_index_display.value].value = self.spinBox_eye_tracking_configuration_radius.value()

        def spinBox_eye_tracking_configuration_angles_valueChanged(self):
            self.shared.eye_tracking_configuration_angles[self.shared.fish_index_display.value].value = self.spinBox_eye_tracking_configuration_angles.value()

        def checkBox_eye_tracking_configuration_display_tracking_process_clicked(self):
            if self.checkBox_eye_tracking_configuration_display_tracking_process.checkState() == QtCore.Qt.Checked:
                self.shared.eye_tracking_configuration_display_tracking_process[self.shared.fish_index_display.value].value = 1
            else:
                self.shared.eye_tracking_configuration_display_tracking_process[self.shared.fish_index_display.value].value = 0

        def spinBox_tail_tracking_set_x0_valueChanged(self):
            self.shared.tail_tracking_set_x0[self.shared.fish_index_display.value].value = self.spinBox_tail_tracking_set_x0.value()

        def spinBox_tail_tracking_set_y0_valueChanged(self):
            self.shared.tail_tracking_set_y0[self.shared.fish_index_display.value].value = self.spinBox_tail_tracking_set_y0.value()

        def spinBox_tail_tracking_set_nodes_valueChanged(self):
            self.shared.tail_tracking_set_nodes[self.shared.fish_index_display.value].value = self.spinBox_tail_tracking_set_nodes.value()

        def doubleSpinBox_tail_tracking_set_bout_start_vigor_valueChanged(self):
            self.shared.tail_tracking_set_bout_start_vigor[self.shared.fish_index_display.value].value = self.doubleSpinBox_tail_tracking_set_bout_start_vigor.value()

        def doubleSpinBox_tail_tracking_set_bout_end_vigor_valueChanged(self):
            self.shared.tail_tracking_set_bout_end_vigor[self.shared.fish_index_display.value].value = self.doubleSpinBox_tail_tracking_set_bout_end_vigor.value()

        def spinBox_experiment_configuration_number_of_trials_valueChanged(self):
            self.shared.experiment_configuration_number_of_trials.value = self.spinBox_experiment_configuration_number_of_trials.value()

        def spinBox_experiment_configuration_trial_time_valueChanged(self):
            self.shared.experiment_configuration_trial_time.value = self.spinBox_experiment_configuration_trial_time.value()

        def lineEdit_experiment_configuration_storage_root_path_textChanged(self):
            fish_index = self.shared.fish_index_display.value

            self.tail_tracking_xs = np.ctypeslib.as_array(self.shared.tail_tracking_xs[fish_index])
            self.tail_tracking_ys = np.ctypeslib.as_array(self.shared.tail_tracking_ys[fish_index])
            self.recorded_tail_tracking_xs = np.ctypeslib.as_array(self.shared.recorded_tail_tracking_xs[fish_index])
            self.recorded_tail_tracking_ys = np.ctypeslib.as_array(self.shared.recorded_tail_tracking_ys[fish_index])
            self.recorded_time = np.ctypeslib.as_array(self.shared.recorded_time[fish_index])

            ## Read in previously recorded data
            text = self.lineEdit_experiment_configuration_storage_root_path.text()#.encode()
            print(text)
            data = np.load(text)

            ## Set correct parameters: num nodes, rois
            maxlen_replay = np.minimum(len(data['tail_shape_xs'].flatten()), 60*12000)
            self.recorded_tail_tracking_xs[:maxlen_replay] = data['tail_shape_xs'].flatten()[:maxlen_replay]
            #print(data['tail_shape_xs'].shape)
            print(fish_index, self.recorded_tail_tracking_xs[:40], data['camera_time'][:5])
            #print(self.shared.recorded_tail_tracking_xs[fish_index][10])
            #print('booo',self.shared.recorded_tail_tracking_xs[fish_index][0,:])
            self.recorded_tail_tracking_ys[:maxlen_replay] = data['tail_shape_ys'].flatten()[:maxlen_replay]
            self.recorded_time[:12000] = data['camera_time'][:12000]
            self.shared.recorded_timeindex[fish_index].value = 0
 
            #self.spinBox_tail_tracking_set_x0.setValue(self.shared.recorded_tail_tracking_xs[fish_index][0,0])
            #self.spinBox_tail_tracking_set_y0.setValue(self.shared.recorded_tail_tracking_ys[fish_index][0,0])

            #nn = self.shared.recorded_tail_tracking_xs[fish_index].shape[1]
            #self.spinBox_tail_tracking_set_nodes.setValue(nn)

            #print('hihihi',self.shared.recorded_tail_tracking_xs[fish_index][0,0])
            self.spinBox_tail_tracking_set_x0.setValue(self.recorded_tail_tracking_xs[0])
            self.spinBox_tail_tracking_set_y0.setValue(self.recorded_tail_tracking_ys[0])
            self.spinBox_tail_tracking_set_nodes.setValue(data['tail_shape_xs'].shape[1])

            ### HERE
            nodes = self.shared.tail_tracking_set_nodes[fish_index].value
            self.tail_tracking_xs[:nodes] = self.recorded_tail_tracking_xs[:nodes].copy()
            self.tail_tracking_ys[:nodes] = self.recorded_tail_tracking_xs[:nodes].copy()
            print(self.shared.recorded_timeindex[fish_index].value, self.tail_tracking_xs)
            print(np.ctypeslib.as_array(self.shared.tail_tracking_xs[fish_index]))
            #self.shared.tail_tracking_xs[fish_index] = np.array([self.shared.recorded_tail_tracking_xs[fish_index][self.shared.recorded_timeindex[fish_index].value][i] for i in range(self.shared.recorded_tail_tracking_xs[fish_index].shape[1])])
            #self.shared.tail_tracking_ys[fish_index] = np.array([self.shared.recorded_tail_tracking_ys[fish_index][self.shared.recorded_timeindex[fish_index].value][i] for i in range(self.shared.recorded_tail_tracking_xs[fish_index].shape[1])])
            

            self.shared.experiment_configuration_storage_root_path[:len(text.encode())] = text.encode()
            self.shared.experiment_configuration_storage_root_path_l.value = len(text.encode())

        def pushButton_experiment_configuration_load_storage_root_path_clicked(self):
            full_path_to_module = os.path.abspath(QtWidgets.QFileDialog.getOpenFileName()[0])
            self.lineEdit_experiment_configuration_storage_root_path.setText(full_path_to_module)

        def pushButton_start_stop_experiment_clicked(self):
            if self.shared.experiment_flow_control_currently_running.value == 0:
                self.shared.experiment_flow_control_start_requested.value = 1
            else:
                self.shared.experiment_flow_control_stop_requested.value = 1

        def checkBox_experiment_configuration_store_head_tail_data_clicked(self):
            if self.checkBox_experiment_configuration_store_head_tail_data.checkState() == QtCore.Qt.Checked:
                self.shared.experiment_configuration_store_head_tail_data.value = 1
            else:
                self.shared.experiment_configuration_store_head_tail_data.value = 0

        def checkBox_experiment_configuration_store_head_tail_movie_clicked(self):
            if self.checkBox_experiment_configuration_store_head_tail_movie.checkState() == QtCore.Qt.Checked:
                self.shared.experiment_configuration_store_head_tail_movie.value = 1
            else:
                self.shared.experiment_configuration_store_head_tail_movie.value = 0

        def keyPressEvent(self, e):
            if e.key() == QtCore.Qt.Key_Escape:
                self.close()

        def update_gui(self):

            fish_index = self.shared.fish_index_display.value

            #######
            # Update the fish image and tail tracking points

            try:
                fish_camera_image = np.ctypeslib.as_array(self.shared.fish_camera_image[fish_index])[:self.shared.fish_camera_image_width[fish_index].value * self.shared.fish_camera_image_height[fish_index].value].copy()
                fish_camera_image.shape = (self.shared.fish_camera_image_width[fish_index].value, self.shared.fish_camera_image_height[fish_index].value)
                #print('hi')
                self.pyqtgraph_fish_camera_display_image_item.setImage(np.fliplr(fish_camera_image.T), autoLevels=False, levels=(0, 255))
                self.pyqtgraph_fish_camera_display_image_item.setRect(QtCore.QRectF(-1, -1, 2, 2))
                self.pyqtgraph_fish_camera_display.setAspectLocked(True, fish_camera_image.shape[1] / fish_camera_image.shape[0])

                # draw the points
                nodes = self.shared.tail_tracking_set_nodes[fish_index].value

                x = 2 * (np.ctypeslib.as_array(self.shared.tail_tracking_xs[fish_index])[:nodes].copy() / fish_camera_image.shape[1]) - 1
                y = 2 * (np.ctypeslib.as_array(self.shared.tail_tracking_ys[fish_index])[:nodes].copy() / fish_camera_image.shape[0]) - 1

                self.pyqtgraph_fish_camera_display_tail_shape_item.setData(x, -y, symbol='o', symbolPen=None, symbolSize=10, symbolBrush=pg.mkBrush(color=(44, 160, 44)))
            except:
                pass
            #print('asdfasdf')
            #######
            # Update the the stimulus information
            stimulus_information = ""
            stimulus_information += "Stimulus name:\t\t{}\n\n".format(bytearray(self.shared.stimulus_information_name[:self.shared.stimulus_information_name_l.value]).decode())
            stimulus_information += "Number of stimuli:\t{}\n".format(self.shared.stimulus_information_number_of_stimuli.value)
            stimulus_information += "Time per stimulus:\t{:.02f} s\n\n".format(self.shared.stimulus_information_time_per_stimulus.value)
            stimulus_information += "Current stimulus index:\t{}\n".format(self.shared.stimulus_flow_control_index[fish_index].value)
            stimulus_information += "Current stimulus time:\t{:.02f} s".format(self.shared.stimulus_flow_control_current_time[fish_index].value)
            self.label_stimulus_information.setText(stimulus_information)

            if self.shared.stimulus_flow_control_currently_running[fish_index].value == 0:
                self.pushButton_stimulus_configuration_start_test_stimulus_index.setText("Start")
            else:
                self.pushButton_stimulus_configuration_start_test_stimulus_index.setText("Stop")

            #######
            # Update the the camera information
            camera_information = "Framerate:\t{:.1f} Hz\nTimestamp:\t{:.2f} s\n\nGain:\t\t{:.2f} dB\nShutter:\t\t{:.2f} ms".format(
                self.shared.fish_camera_fps[fish_index].value,
                self.shared.fish_camera_timestamp[fish_index].value,
                self.shared.fish_camera_gain[fish_index].value,
                self.shared.fish_camera_shutter[fish_index].value)
            self.label_camera_information.setText(camera_information)

            head_tail_tracking_information = "Left eye angle:\t\t{:.1f}°\nRight eye angle:\t\t{:.1f}°\n\nTail tip deflection: \t{:.1f}°".format(
                self.shared.eye_tracking_circular_history_angle_left_eye[fish_index][self.shared.tail_tracking_circular_counter[fish_index].value],
                self.shared.eye_tracking_circular_history_angle_right_eye[fish_index][self.shared.tail_tracking_circular_counter[fish_index].value],
                self.shared.tail_tracking_circular_history_tail_tip_deflection[fish_index][self.shared.tail_tracking_circular_counter[fish_index].value])
            self.label_head_tail_tracking_information.setText(head_tail_tracking_information)

            ######
            # Update the experiment information
            total_time = self.shared.experiment_configuration_number_of_trials.value * self.shared.experiment_configuration_trial_time.value / 60.

            currently_storing_head_tail_data = np.array([self.shared.experiment_flow_control_currently_storing_head_tail_data[fish_index].value == 1 for fish_index in range(4)]).any()
            currently_storing_stimulus_data = np.array([self.shared.experiment_flow_control_currently_storing_stimulus_data[fish_index].value == 1 for fish_index in range(4)]).any()

            experiment_information = "Estimated total time:\t{:.1f} min\n\nCurrent trial:\t\t{}\nCurrently storing data:\t{}".format(total_time,
                                                                                                                                 self.shared.experiment_flow_control_current_trial.value,
                                                                                                                                 currently_storing_head_tail_data == True or currently_storing_stimulus_data == True)
            self.label_experiment_information.setText(experiment_information)

            self.progressBar_experiment_flow_control_percentage_done.setValue(int(self.shared.experiment_flow_control_percentage_done.value))

            if self.shared.experiment_flow_control_currently_running.value == 1:
                stylesheet = """
                    .QPushButton {
                        background-color: red; color:black; border-style: outset;
                        border-width: 2px;
                        border-color: darkred;
                        }
                        """

                self.pushButton_start_stop_experiment.setStyleSheet(stylesheet)
                self.pushButton_start_stop_experiment.setText("Stop experiment")

            if self.shared.experiment_flow_control_currently_running.value == 0:
                stylesheet = """
                                    .QPushButton {
                                        background-color: green; color:black; border-style: outset;
                                        border-width: 2px;
                                        border-color: darkgreen;
                                        }
                                        """

                self.pushButton_start_stop_experiment.setStyleSheet(stylesheet)
                self.pushButton_start_stop_experiment.setText("Start experiment")

        def closeEvent(self, event):

            self.save_program_configuration()

            if self.shared.experiment_flow_control_currently_running.value == 1:
                self.shared.experiment_flow_control_stop_requested.value == 1

            self.shared.running.value = 0

            self.close()


    app = QtWidgets.QApplication(sys.argv)

    embedded_4fish_gui = Embedded_4fish_gui()

    embedded_4fish_gui.show()
    app.exec_()

    shared.running.value = 0
