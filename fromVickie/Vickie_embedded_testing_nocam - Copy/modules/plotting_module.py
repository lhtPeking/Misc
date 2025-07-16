from multiprocessing import Process
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5 import QtWidgets
import numpy as np
from scipy.interpolate import interp1d

class PlottingModule(Process):
    def __init__(self, shared):
        Process.__init__(self)

        self.shared = shared

    def run(self):

        self.frozen = False

        pg.setConfigOption('background', pg.mkColor(20 / 255.))
        pg.setConfigOption('foreground', 'w')
        pg.setConfigOptions(antialias=False) # antialialsoung makes it very slow

        self.app = QtWidgets.QApplication([])

        self.view = pg.GraphicsView()
        self.layout = pg.GraphicsLayout()
        self.layout.layout.setSpacing(15.)
        self.layout.setContentsMargins(10., 10., 10., 10.)
        self.view.setCentralItem(self.layout)

        self.view.keyPressEvent = self.keyPressEvent

        self.view.resize(1000, 1000)

        # make some frozen buffers
        self.tail_tracking_circular_counter_frozen = [0 for _ in range(4)]
        self.tail_tracking_circular_history_time_frozen = [[] for _ in range(4)]

        self.eye_tracking_circular_history_angle_left_eye_frozen = [[] for _ in range(4)]
        self.eye_tracking_circular_history_angle_right_eye_frozen = [[] for _ in range(4)]

        self.tail_tracking_circular_history_tail_tip_deflection_frozen = [[] for _ in range(4)]
        self.tail_tracking_circular_history_tail_tip_deflection_sliding_window_variance_frozen = [[] for _ in range(4)]
        self.tail_tracking_circular_history_bout_information_frozen = [[] for _ in range(4)]

        self.plot_widgets = []

        names = ["Left eye (°)", "Right eye (°)", "Tail tip deflection (°)", "Tail vigor"]
        labelStyle = {'color': '#FFF', 'font-size': '{}pt'.format(10)}

        self.infinitelines = []

        for i in range(4):
            self.plot_widgets.append(self.layout.addPlot(i, 0))
            self.plot_widgets[i].setXRange(0, 30) # 0 to 30 seconds only
            self.plot_widgets[i].setLabel('left', names[i], **labelStyle)
            self.plot_widgets[i].showGrid(x=True, y=True)
            self.plot_widgets[i].getAxis('left').setWidth(w=100)

            self.infinitelines.append(pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(color=(255, 0, 0, 128), width=2.5), movable=True, bounds=None))

            if i == 3:
                self.plot_widgets[i].setLabel('bottom', "Time (s)", **labelStyle)

            if i < 3:
                self.plot_widgets[i].getAxis('bottom').setStyle(showValues=False)

            if i > 0:
                self.plot_widgets[i].setXLink(self.plot_widgets[0])

        self.infinitelines[0].sigPositionChangeFinished.connect(self.update_infiniteline_0)
        self.infinitelines[1].sigPositionChangeFinished.connect(self.update_infiniteline_1)
        self.infinitelines[2].sigPositionChangeFinished.connect(self.update_infiniteline_2)
        self.infinitelines[3].sigPositionChangeFinished.connect(self.update_infiniteline_3)

        self.infinitelines[0].sigPositionChanged.connect(self.update_infiniteline_0)
        self.infinitelines[1].sigPositionChanged.connect(self.update_infiniteline_1)
        self.infinitelines[2].sigPositionChanged.connect(self.update_infiniteline_2)
        self.infinitelines[3].sigPositionChanged.connect(self.update_infiniteline_3)

        # data items
        self.left_eye_data = pg.PlotDataItem(pen=pg.mkPen(color=(31, 119, 180), width=2.5))
        self.right_eye_data = pg.PlotDataItem(pen=pg.mkPen(color=(255, 127, 14), width=2.5))

        self.tail_tip_deflection_data = pg.PlotDataItem(pen=pg.mkPen(color=(44, 160, 44), width=2.5))
        self.tail_vigor_data = pg.PlotDataItem(pen=pg.mkPen(color=(44, 160, 44), width=2.5))
        self.tail_vigor_bout_starts_data = pg.PlotDataItem(symbol='o', pen=None,
                                                                    symbolBrush=pg.mkBrush(color=(255, 0, 0)),
                                                                    symbolPen=None,
                                                                    symbolSize=10)

        self.tail_vigor_bout_ends_data = pg.PlotDataItem(symbol='o', pen=None,
                                                                  symbolBrush=pg.mkBrush(color=(0, 255, 0)),
                                                                  symbolPen=None,
                                                                  symbolSize=10)

        self.plot_widgets[0].addItem(self.left_eye_data)
        self.plot_widgets[1].addItem(self.right_eye_data)

        self.plot_widgets[2].addItem(self.tail_tip_deflection_data)

        self.plot_widgets[3].addItem(self.tail_vigor_data)
        self.plot_widgets[3].addItem(self.tail_vigor_bout_starts_data)
        self.plot_widgets[3].addItem(self.tail_vigor_bout_ends_data)


        self.plot_widgets[0].setYRange(0, 180)  # left eye
        self.plot_widgets[1].setYRange(0, 180)  # right eye
        self.plot_widgets[2].setYRange(-200, 200)  # tail angle
        self.plot_widgets[3].setYRange(0, 10000)  # tail vigor

        self.plot_widgets[0].setLimits(xMin=0, xMax=30, yMin=0, yMax=180)
        self.plot_widgets[1].setLimits(xMin=0, xMax=30, yMin=0, yMax=180)
        self.plot_widgets[2].setLimits(xMin=0, xMax=30, yMin=-200, yMax=200)
        self.plot_widgets[3].setLimits(xMin=0, xMax=30, yMin=0, yMax=10000)

        # on to the helper lines
        for i in range(4):
            self.plot_widgets[i].addItem(self.infinitelines[i])

        self.update_gui_timer = QtCore.QTimer()
        self.update_gui_timer.timeout.connect(self.update_gui)
        self.update_gui_timer.start(20)

        self.view.show()

        QtWidgets.QApplication.instance().exec_()

    def update_infiniteline_0(self):
        for i in range(4):
            self.infinitelines[i].setValue(self.infinitelines[0].value())

    def update_infiniteline_1(self):
        for i in range(4):
            self.infinitelines[i].setValue(self.infinitelines[1].value())

    def update_infiniteline_2(self):
        for i in range(4):
            self.infinitelines[i].setValue(self.infinitelines[2].value())

    def update_infiniteline_3(self):
        for i in range(4):
            self.infinitelines[i].setValue(self.infinitelines[3].value())


    def keyPressEvent(self, e):

        if e.key() == QtCore.Qt.Key_Space:
            # if frozen, free the data
            if self.frozen == True:
                self.frozen = False
            else:
                for fish_index in range(4):
                    self.tail_tracking_circular_counter_frozen[fish_index] = self.shared.tail_tracking_circular_counter[fish_index].value

                    self.tail_tracking_circular_history_time_frozen[fish_index] = np.ctypeslib.as_array(self.shared.tail_tracking_circular_history_time[fish_index]).copy()

                    self.eye_tracking_circular_history_angle_left_eye_frozen[fish_index] = np.ctypeslib.as_array(self.shared.eye_tracking_circular_history_angle_left_eye[fish_index]).copy()
                    self.eye_tracking_circular_history_angle_right_eye_frozen[fish_index] = np.ctypeslib.as_array(self.shared.eye_tracking_circular_history_angle_right_eye[fish_index]).copy()

                    self.tail_tracking_circular_history_tail_tip_deflection_frozen[fish_index] = np.ctypeslib.as_array(self.shared.tail_tracking_circular_history_tail_tip_deflection[fish_index]).copy()
                    self.tail_tracking_circular_history_tail_tip_deflection_sliding_window_variance_frozen[fish_index] = np.ctypeslib.as_array(self.shared.tail_tracking_circular_history_tail_tip_deflection_sliding_window_variance[fish_index]).copy()
                    self.tail_tracking_circular_history_bout_information_frozen[fish_index] = np.ctypeslib.as_array(self.shared.tail_tracking_circular_history_bout_information[fish_index]).copy()

                self.frozen = True

    def update_gui(self):

        fish_index = self.shared.fish_index_display.value

        x_viewrange = self.plot_widgets[0].viewRange()[0]

        if self.frozen == False:
            tail_tracking_circular_counter = self.shared.tail_tracking_circular_counter[fish_index].value

            tail_tracking_circular_history_time = np.ctypeslib.as_array(self.shared.tail_tracking_circular_history_time[fish_index])

            eye_tracking_circular_history_angle_left_eye = np.ctypeslib.as_array(self.shared.eye_tracking_circular_history_angle_left_eye[fish_index])
            eye_tracking_circular_history_angle_right_eye = np.ctypeslib.as_array(self.shared.eye_tracking_circular_history_angle_right_eye[fish_index])

            tail_tracking_circular_history_tail_tip_deflection = np.ctypeslib.as_array(self.shared.tail_tracking_circular_history_tail_tip_deflection[fish_index])
            tail_tracking_circular_history_tail_tip_deflection_sliding_window_variance = np.ctypeslib.as_array(self.shared.tail_tracking_circular_history_tail_tip_deflection_sliding_window_variance[fish_index])
            tail_tracking_circular_history_bout_information = np.ctypeslib.as_array(self.shared.tail_tracking_circular_history_bout_information[fish_index])

        else:
            tail_tracking_circular_counter = self.tail_tracking_circular_counter_frozen[fish_index]

            tail_tracking_circular_history_time = self.tail_tracking_circular_history_time_frozen[fish_index]

            eye_tracking_circular_history_angle_left_eye = self.eye_tracking_circular_history_angle_left_eye_frozen[fish_index]
            eye_tracking_circular_history_angle_right_eye = self.eye_tracking_circular_history_angle_right_eye_frozen[fish_index]

            tail_tracking_circular_history_tail_tip_deflection = self.tail_tracking_circular_history_tail_tip_deflection_frozen[fish_index]
            tail_tracking_circular_history_tail_tip_deflection_sliding_window_variance = self.tail_tracking_circular_history_tail_tip_deflection_sliding_window_variance_frozen[fish_index]
            tail_tracking_circular_history_bout_information = self.tail_tracking_circular_history_bout_information_frozen[fish_index]

        # make circular buffers right
        tail_time = np.r_[tail_tracking_circular_history_time[tail_tracking_circular_counter + 1:], tail_tracking_circular_history_time[:tail_tracking_circular_counter]]

        eye_angle_left = np.r_[eye_tracking_circular_history_angle_left_eye[tail_tracking_circular_counter + 1:], eye_tracking_circular_history_angle_left_eye[:tail_tracking_circular_counter]]
        eye_angle_right = np.r_[eye_tracking_circular_history_angle_right_eye[tail_tracking_circular_counter + 1:], eye_tracking_circular_history_angle_right_eye[:tail_tracking_circular_counter]]

        tail_angle = np.r_[tail_tracking_circular_history_tail_tip_deflection[tail_tracking_circular_counter + 1:], tail_tracking_circular_history_tail_tip_deflection[:tail_tracking_circular_counter]]
        tail_sliding_window_variances = np.r_[tail_tracking_circular_history_tail_tip_deflection_sliding_window_variance[tail_tracking_circular_counter + 1:], tail_tracking_circular_history_tail_tip_deflection_sliding_window_variance[:tail_tracking_circular_counter]]
        bout_info = np.r_[tail_tracking_circular_history_bout_information[tail_tracking_circular_counter + 1:], tail_tracking_circular_history_bout_information[:tail_tracking_circular_counter]]

        # the is 30s in the past, make everything relative to that
        t0 = tail_time[-1] - 30

        # only plot the last 30s
        ind = np.where((tail_time >= t0 + x_viewrange[0]) & (tail_time <= t0 + x_viewrange[1]))

        if len(ind[0]) > 0:
            tail_time_view = tail_time[ind]

            eye_angle_left_view = eye_angle_left[ind]
            eye_angle_right_view = eye_angle_right[ind]

            tail_angle_view = tail_angle[ind]
            tail_sliding_window_variances_view = tail_sliding_window_variances[ind]
            bout_info_view = bout_info[ind]

            # find bout starts
            ind = np.where(bout_info_view == 1)

            bout_starts_time = tail_time_view[ind]
            bout_starts_vigor = tail_sliding_window_variances_view[ind]

            # find bout ends
            ind = np.where(bout_info_view == 3)
            bout_ends_time = tail_time_view[ind]
            bout_ends_vigor = tail_sliding_window_variances_view[ind]

            if len(tail_time_view) > 1000: # if too many data points in the view range, downsample

                new_t = np.linspace(t0 + x_viewrange[0], t0 + x_viewrange[1], 1000)

                f = interp1d(tail_time_view, eye_angle_left_view, bounds_error=False, fill_value=np.nan)
                eye_angle_left_view = f(new_t)

                f = interp1d(tail_time_view, eye_angle_right_view, bounds_error=False, fill_value=np.nan)
                eye_angle_right_view = f(new_t)

                f = interp1d(tail_time_view, tail_angle_view, bounds_error=False, fill_value=np.nan)
                tail_angle_view = f(new_t)

                f = interp1d(tail_time_view, tail_sliding_window_variances_view, bounds_error=False, fill_value=np.nan)
                tail_sliding_window_variances_view = f(new_t)
            else:
                new_t = tail_time_view

            # left eye angle
            self.left_eye_data.setData(new_t - t0, eye_angle_left_view)

            # right eye angle
            self.right_eye_data.setData(new_t - t0, eye_angle_right_view)

            # tail tip deflection
            self.tail_tip_deflection_data.setData(new_t - t0, tail_angle_view)

            # tail vigor
            self.tail_vigor_data.setData(new_t - t0, tail_sliding_window_variances_view)

            # bouts
            if len(bout_starts_time) > 0:
                self.tail_vigor_bout_starts_data.setData(bout_starts_time - t0, bout_starts_vigor)
            else:
                self.tail_vigor_bout_starts_data.setData([], [])

            if len(bout_ends_time) > 0:
                self.tail_vigor_bout_ends_data.setData(bout_ends_time - t0, bout_ends_vigor)
            else:
                self.tail_vigor_bout_ends_data.setData([], [])

        if self.shared.running.value == 0:
            self.view.close()
