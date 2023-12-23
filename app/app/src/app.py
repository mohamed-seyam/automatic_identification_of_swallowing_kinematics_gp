import os
import sys

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QThread, QObject, pyqtSignal


from app.modules.opticalflow import OpticalFlow
from app.modules.dynamic_frame_classification import DynamicFramesClassifier
from app.modules.frame_extraction import FramesExtractor
from app.modules.pharyngeal_classifier import PharyngealClassifier
from app.modules.bolus_segmentation import BolusSegmenter
from app.modules.hyoid_bone_displacement import HyoidBoneLocalizer

from hyoidboneDisplacement import get_hdboneDisplacement

from utils import *
from gui import Ui_MainWindow

from pathlib import Path

import pandas as pd
import numpy as np
import cv2

# COMMON PATHS USED IN THE CODE
APPLICATION_DIRECTORY           = os.getcwd()
DATA_PATH                       = os.path.join(APPLICATION_DIRECTORY, "data")

PAUSED = False
PLAYING = True

hyoid = False

class VideoCapture(QObject):
    play_state_changed = pyqtSignal(bool)
    frameChanged = pyqtSignal(int)

    def __init__(self, video_dir, labelWidget, stateWidget, scrollWidget, FPS = 30):
        super().__init__()
        self.cap = cv2.VideoCapture(video_dir)
        self.video_frame = labelWidget
        self.fps = FPS
        self.totalFrameCount = int(self.cap.get(cv2. CAP_PROP_FRAME_COUNT))
        self.durationInMS = (1 / 30) * self.totalFrameCount
        self.timer = QtCore.QTimer()
        self.stateWidget = stateWidget

        self.timer.setInterval(int(1000/self.fps))
        self.timer.start()
        self.video_dir = video_dir
        self.currentFrame = None
        self.currentFrameIdx = 0
        self.Framestate = str(self.currentFrameIdx)
        self.stateWidget.setText(self.Framestate)

        self.scrollWidget = scrollWidget
        self.scrollWidget.setRange(0, self.totalFrameCount)
        self.scrollWidget.setValue(0)

        self.brightness = 0
        self.contrast = 1

        self.hyoidbone_enable       = False
        self.bolus_mask_enable      = False
        self.c2_c4_enable           = False
        self.hyoid_data_available   = False
        self.c2_c4_data_available   = False
        self.bolus_data_available   = False

        self.hyoid_data_dir = None
        self.c2_c4_data_dir = None
        self.bolus_data_dir = None


        self.play_state = False
        self.nextFrameSlot()
        self.timer.timeout.connect(self.move)

    def move(self):
        if self.play_state != PAUSED:
            self.nextFrameSlot()

    def nextFrameSlot(self):
        if self.currentFrameIdx < self.totalFrameCount:
            self.displayNextFrame()
            self.currentFrameIdx = self.currentFrameIdx + 1
            self.Framestate = str(self.currentFrameIdx)
            self.stateWidget.setText(self.Framestate)
            self.scrollWidget.setValue(self.currentFrameIdx - 1)

    def apply_hyoid_detection(self):
        if self.hyoidbone_enable and self.hyoid_data_available:
            draw_bounding_box_on_image( self.currentFrame,
                                        self.hyoid_data_dir.loc[self.currentFrameIdx, 'ymin'],
                                        self.hyoid_data_dir.loc[self.currentFrameIdx, 'xmin'],
                                        self.hyoid_data_dir.loc[self.currentFrameIdx, 'ymax'],
                                        self.hyoid_data_dir.loc[self.currentFrameIdx, 'xmax'],
                                        "Hyoid",
                                        color='red',
                                        thickness=2,
                                        use_normalized_coordinates=True)

    def apply_bolus_segmentation(self):
        if self.bolus_mask_enable and self.bolus_data_available:
            try:
                mask_img = cv2.imread(os.path.join(self.bolus_data_dir, f"{self.currentFrameIdx}.png"), 0)
                mask_img = cv2.resize(mask_img,(512,512),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
                self.currentFrame = np.array(cv2.cvtColor(self.currentFrame, cv2.COLOR_BGR2GRAY))
                self.currentFrame = cv2.resize(self.currentFrame,(512,512),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
                self.currentFrame = overlay_mask(self.currentFrame, mask_img)
            except:
                print("Error in bolus mask overlaying!")

    def apply_c2_c4_detection(self):
        if self.c2_c4_enable and self.c2_c4_data_available:
            draw_bounding_box_on_image( self.currentFrame,
                                        self.c2_c4_data_dir.loc[self.currentFrameIdx, 'ymin'],
                                        self.c2_c4_data_dir.loc[self.currentFrameIdx, 'xmin'],
                                        self.c2_c4_data_dir.loc[self.currentFrameIdx, 'ymax'],
                                        self.c2_c4_data_dir.loc[self.currentFrameIdx, 'xmax'],
                                        "C2-C4",
                                        color='blue',
                                        thickness=2,
                                        use_normalized_coordinates=True)

    def displayNextFrame(self):
        ret, self.currentFrame_original = self.cap.read()
        self.currentFrame = self.currentFrame_original
        self.currentFrame = cv2.convertScaleAbs(self.currentFrame_original, alpha = self.contrast, beta = self.brightness)
        self.apply_bolus_segmentation()
        self.apply_hyoid_detection()
        self.apply_c2_c4_detection()

        frame_resized = cv2.resize(self.currentFrame,(400,450),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        qimg = toQImage(frame_resized, copy = True)
        pix = QPixmap(qimg)
        self.video_frame.setPixmap(pix)
        self.frameChanged.emit(self.currentFrameIdx)

    def displayCurrentFrame(self):
        self.currentFrame = cv2.convertScaleAbs(self.currentFrame_original, alpha = self.contrast, beta = self.brightness)
        self.apply_bolus_segmentation()
        self.apply_hyoid_detection()
        self.apply_c2_c4_detection()

        frame_resized = cv2.resize(self.currentFrame,(400,450),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        qimg = toQImage(frame_resized, copy = True)
        pix = QPixmap(qimg)
        self.video_frame.setPixmap(pix)

    def setFrameIndex(self, frameIdx):
        if self.totalFrameCount < frameIdx or frameIdx < 0:
            return
        else:
            self.currentFrameIdx = frameIdx
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frameIdx)
            self.nextFrameSlot()

    def pause_resume(self):
        self.play_state = not self.play_state
        self.play_state_changed.emit(not self.play_state)

    def get_curr_frame_idx(self):
        return self.currentFrameIdx

    def save(self):
        self.setFrameIndex(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi',fourcc, 5, (1280,720))

    def setContrast(self, contrast):
        self.contrast = contrast / 100
        self.displayCurrentFrame()

    def setfps(self, fps):
        self.fps = fps
        self.timer.stop()
        self.timer.setInterval(int(1000/self.fps))
        self.timer.start()

    def setbrightness(self, perc):
        self.brightness = perc
        self.displayCurrentFrame()

class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.capture = None
        self.video_dir = None
        self.video_name = None
        self.isVideoFileLoaded = False
        self.frame_names = None
        self.threadsNum = 0
        self.threads = {}
        self.ui.inlineReport.cellClicked.connect(self.go_to_frame_from_table_s_d)
        self.ui.inlineReport_2.cellClicked.connect(self.go_to_frame_from_table_pharyngeal)
        self.ui.displacement_options_cb.currentIndexChanged.connect(self.displacementOptionChanged)
        self.displacement_option = "None"
        self.curr_region_start = None
        self.curr_region_end = None

        # Link the UI components with the events
        self.ui.openButton.clicked.connect(self.loadVideoFile)
        # self.ui.goButton.clicked.connect(self.jumpFrame)
        self.ui.nextFrameButton.clicked.connect(self.nextFrame)
        self.ui.prevFrameButton.clicked.connect(self.prevFrame)
        self.ui.contrastSlider.valueChanged.connect(self.updateContrast)
        self.ui.brightnessSlider.valueChanged.connect(self.updateBrightness)
        self.ui.fpsSlider.valueChanged.connect(self.updateFps)
        self.ui.scrollBar.valueChanged.connect(self.scrollFrame)
        self.ui.pause_resume_button.setIcon(QtGui.QIcon(QtGui.QPixmap("media/play-button-icon.jpg")))

    def displacementOptionChanged(self, idx):
        self.displacement_option = self.ui.displacement_options_cb.currentText()
        selected_frame_indecies = [idx for idx in range(self.curr_region_start, self.curr_region_end+1)]
        img = get_hdboneDisplacement(selected_frame_indecies, self.hyoid_data_dir, self.c2_c4_data_dir, self.video_data_dir, displacement_option=self.displacement_option)
        if img is not None:
            img = cv2.resize(img,(400,250),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
            self.ui.hdbone_displacement_output_img.setPixmap(cv2_to_pixmap(img))
            self.ui.hdbone_displacement_output_img.show()
        else:
            self.ui.hdbone_displacement_output_img.hide()

    def go_to_frame_from_table_s_d(self, row, col):
        if col < 2:
            item_selected = self.ui.inlineReport.item(row, col).text()
            self.capture.setFrameIndex(int(item_selected))

    def go_to_frame_from_table_pharyngeal(self, row, col):
        if col < 2:
            item_selected = self.ui.inlineReport_2.item(row, col).text()
            self.capture.setFrameIndex(int(item_selected))
            start = int(self.ui.inlineReport_2.item(row, 0).text())
            end = int(self.ui.inlineReport_2.item(row, 1).text())
            self.curr_region_start = start
            self.curr_region_end = end
            selected_frame_indecies = [idx for idx in range(start, end+1)]
            img = get_hdboneDisplacement(selected_frame_indecies, self.hyoid_data_dir, self.c2_c4_data_dir, self.video_data_dir, displacement_option=self.displacement_option)
            if img is not None:
                img = cv2.resize(img,(400,250),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
                self.ui.hdbone_displacement_output_img.setPixmap(cv2_to_pixmap(img))
                self.ui.hdbone_displacement_output_img.show()
            else:
                self.ui.hdbone_displacement_output_img.hide()

    def updateContrast(self):
        if self.capture:
            self.capture.setContrast(self.ui.contrastSlider.value())

    def updateBrightness(self):
        if self.capture:
            self.capture.setbrightness(self.ui.brightnessSlider.value())

    def updateFps(self):
        if self.capture:
            self.capture.setfps(self.ui.fpsSlider.value())

    def scrollFrame(self):
        if self.capture:
            self.capture.setFrameIndex(self.ui.scrollBar.value())

    def jumpFrame(self):
        if self.capture:
            try:
                frameIndexInput = self.ui.frameLineEdit.text()
                self.capture.setFrameIndex(int(frameIndexInput))
            except:
                print("not a valid frame index!") ######### warning

    def createThread(self, worker, threadName, finish_callback_function=None, finish_callback_function_2=None):

        # Create a QThread object
        self.threads[threadName] = QThread()
        self.loadingON()
        self.ui.runningProcessLabel.setText(threadName)
        self.threadsNum = self.threadsNum + 1
        
        # Move worker to the thread
        worker.moveToThread(self.threads[threadName])
        
        # Connect signals and slots
        self.threads[threadName].started.connect(worker.run)
        worker.finished.connect(self.threads[threadName].quit)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(self.loadingOFF)
        worker.progress.connect(self.reportProgress)
        self.threads[threadName].finished.connect(self.threads[threadName].deleteLater)
        if finish_callback_function:
            self.threads[threadName].finished.connect(finish_callback_function)
        if finish_callback_function_2:
            self.threads[threadName].finished.connect(finish_callback_function_2)

        # Start the thread
        print(f"{threadName} thread STARTED,  Number of Active threads: {self.threadsNum}")
        self.threads[threadName].start()

    def start_static_and_dynamic_classification_thread(self):
        self.createThread(self.DynamicClassifierWorker, threadName= 'dynamic and static classification',
                        finish_callback_function=self.setupStaticDynamicDistribution,
                        finish_callback_function_2=self.start_pharyngeal_classification_thread)

    def start_pharyngeal_classification_thread(self):
        self.createThread(self.PharyngealClassifierWorker, threadName= 'pharyngeal classification',
                        finish_callback_function=self.setupPharyngealDistribution,
                        finish_callback_function_2=self.start_bolus_segmentation_thread)

    def start_bolus_segmentation_thread(self):
        if not Path(self.bolus_data_dir).exists():
            mkdir_ifnotexists(self.bolus_data_dir)
            self.createThread(self.bolusSegmentationWorker, threadName='bolus segmentation',
                            finish_callback_function=self.load_bolus_data)

    def setupStaticDynamicDistribution(self):
        video_size = len(os.listdir(self.optflow_dir))

        ######################
        img = create_distribution_img(video_size, self.video_data_dir, self.dynamic_and_static_data_dir_ground_truth, [0, 255, 0], "Ground truth", "distribution_ground_truth.jpg")
        self.ui.s_d_gnd_truth_dist.setPixmap(cv2_to_pixmap(img))

        img = create_distribution_img(video_size, self.video_data_dir, self.dynamic_and_static_data_dir_after_filteration, [0,0,255], "Filtered", "distribution_after_filter.jpg")
        self.ui.s_d_after_filter_dist.setPixmap(cv2_to_pixmap(img))
        ########################
        self.distribution_img = np.ones((video_size, 200, 3)) * 255
        self.target_regions = pd.read_csv(self.dynamic_and_static_data_dir_after_filteration)
        for begin, end in zip(self.target_regions['begin'], self.target_regions['end']):
            self.distribution_img[begin: end, :] = [0, 0, 255]
        self.capture.frameChanged.connect(self.update_current_index_in_dynamic_distribution_image)
        self.loadDataToTable(self.target_regions, table_type="s_d")

    def setupPharyngealDistribution(self):
        video_size = len(os.listdir(self.optflow_dir))

        ######################
        img = create_distribution_img(video_size, self.video_data_dir, self.dynamic_and_static_data_dir_ground_truth, [0, 255, 0], "Ground truth", "distribution_ground_truth.jpg")
        self.ui.s_d_gnd_truth_dist.setPixmap(cv2_to_pixmap(img))

        img = create_distribution_img(video_size, self.video_data_dir, self.dynamic_and_static_data_dir_after_filteration, [0,0,255], "Filtered", "distribution_after_filter.jpg")
        self.ui.s_d_after_filter_dist.setPixmap(cv2_to_pixmap(img))

        img = create_distribution_img(video_size, self.video_data_dir, self.pharyngeal_data_dir_ground_truth, [0, 255, 0], "Ground truth", "pharyngeal_distribution_ground_truth.jpg")
        self.ui.pharyngeal_gnd_truth_dist.setPixmap(cv2_to_pixmap(img))

        img = create_distribution_img(video_size, self.video_data_dir, self.pharyngeal_data_dir_after_filteration, [0,0,255], "Filtered", "pharyngeal_distribution_after_filter.jpg")
        self.ui.pharyngeal_after_filter_dist.setPixmap(cv2_to_pixmap(img))
        ########################
        self.distribution_img = np.ones((video_size, 200, 3)) * 255
        self.target_regions = pd.read_csv(self.pharyngeal_data_dir_ground_truth)
        for begin, end in zip(self.target_regions['begin'], self.target_regions['end']):
            self.distribution_img[begin: end, :] = [0, 255, 0]
        self.capture.frameChanged.connect(self.update_current_index_in_dynamic_distribution_image)
        self.loadDataToTable(self.target_regions, table_type="pharyngeal")

    def update_current_index_in_dynamic_distribution_image(self, idx):
        begin = idx - 5 if idx >= 5 else idx
        end = idx + 5 if idx <= self.capture.totalFrameCount - 5 else idx
        img = self.distribution_img.copy()
        img[begin: end, :] = [0, 0, 0]
        img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.resize(img,(400,7),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        qimg = toQImage(img, copy = True)
        pix = QPixmap(qimg)
        self.ui.pharyngeal_gnd_truth_dist.setPixmap(pix)

    def loadDataToTable(self, dataframe, table_type="s_d"):
        if table_type == "s_d":
            self.ui.inlineReport.show()
            n_rows = dataframe.shape[0]
            self.ui.inlineReport.setColumnCount(3)
            self.ui.inlineReport.setRowCount(n_rows)
            for row in range(n_rows):
                self.ui.inlineReport.setItem(row, 0, QtWidgets.QTableWidgetItem(str(dataframe.loc[row, 'begin'])))
                self.ui.inlineReport.setItem(row, 1, QtWidgets.QTableWidgetItem(str(dataframe.loc[row, 'end'])))
        elif table_type == "pharyngeal":
            self.ui.inlineReport_2.show()
            n_rows = dataframe.shape[0]
            self.ui.inlineReport_2.setColumnCount(3)
            self.ui.inlineReport_2.setRowCount(n_rows)
            for row in range(n_rows):
                self.ui.inlineReport_2.setItem(row, 0, QtWidgets.QTableWidgetItem(str(dataframe.loc[row, 'begin'])))
                self.ui.inlineReport_2.setItem(row, 1, QtWidgets.QTableWidgetItem(str(dataframe.loc[row, 'end'])))

    def reportProgress(self, idx):
        pass

    def nextFrame(self):
        if self.capture:
            self.capture.setFrameIndex(self.capture.currentFrameIdx)

    def prevFrame(self):
        if self.capture:
            self.capture.setFrameIndex(self.capture.currentFrameIdx - 2)

    def startCapture(self):
        if self.isVideoFileLoaded:
            self.capture = VideoCapture(    self.video_dir,
                                            self.ui.video, 
                                            self.ui.framesLabel, 
                                            self.ui.scrollBar)
            self.ui.pause_resume_button.clicked.connect(self.capture.pause_resume)
            self.capture.play_state_changed.connect(self.change_play_icon)
            self.ui.hyoidTracking.stateChanged.connect(self.hyoidTrackingCheckBox)
            self.ui.c2_c4_tracking.stateChanged.connect(self.c2c4TrackingCheckBox)
            self.ui.bolusTracking.stateChanged.connect(self.bolusCheckBox)

    def change_play_icon(self, playing):
        if playing:
            self.ui.pause_resume_button.setIcon(QtGui.QIcon(QtGui.QPixmap("media/play-button-icon.jpg")))
            self.ui.pause_resume_button.setText("Play")
        else:
            self.ui.pause_resume_button.setIcon(QtGui.QIcon(QtGui.QPixmap("media/pause-icon.png")))
            self.ui.pause_resume_button.setText("Pause")

    def endCapture(self):
        self.capture.deleteLater()
        self.capture = None

    def loadingON(self):
        self.ui.loaderGIF.show()
        self.ui.loaderMovie.start()

    def loadingOFF(self):
        self.threadsNum = self.threadsNum - 1
        if self.threadsNum <= 0:
            self.ui.loaderGIF.hide()
            self.ui.runningProcessLabel.setText("")

    def loadVideoFile(self):
        files_name = QtWidgets.QFileDialog.getOpenFileName( self, 'Open only avi', os.getenv('HOME'), "avi(*.avi)" )
        if len(files_name[0]) > 0:
            self.video_dir = files_name[0]
            self.video_name = self.video_dir.split('/')[-1].split('.')[0].lower()
            self.ui.pathTextEdit.setText(self.video_dir)
            self.video_data_dir             = os.path.join(DATA_PATH, self.video_name)
            self.frames_dir                 = os.path.join(DATA_PATH, self.video_name, 'frames')
            self.optflow_dir                = os.path.join(DATA_PATH, self.video_name, 'optflow')
            self.hyoid_data_dir             = os.path.join(DATA_PATH, self.video_name, "hyoidbone.csv")
            self.c2_c4_data_dir             = os.path.join(DATA_PATH, self.video_name, "c2_c4.csv")
            self.bolus_data_dir             = os.path.join(DATA_PATH, self.video_name, "bolus")
            self.dynamic_and_static_data_dir= os.path.join(DATA_PATH, self.video_name, "dynamic_regions.csv")
            self.dynamic_and_static_data_dir_after_filteration= os.path.join(DATA_PATH, self.video_name, "dynamic_regions_filtered.csv")
            self.dynamic_and_static_data_dir_ground_truth= os.path.join(DATA_PATH, self.video_name, "dynamic_regions_ground_truth.csv")
            self.pharyngeal_data_dir        = os.path.join(DATA_PATH, self.video_name, "pharyngeal_regions.csv")
            self.pharyngeal_data_dir_after_filteration  = os.path.join(DATA_PATH, self.video_name, "pharyngeal_regions_after_filteration.csv")
            self.pharyngeal_data_dir_ground_truth       = os.path.join(DATA_PATH, self.video_name, "pharyngeal_regions_ground_truth.csv")

            self.isVideoFileLoaded = True
            self.startCapture()

            mkdir_ifnotexists(self.video_dir)
            
            self.framesExtractorWorker = FramesExtractor(   
                                    frames_dir= self.frames_dir, 
                                    video_dir=self.video_dir)
            self.opticalFlowWorker = OpticalFlow(
                                    video_dir=self.video_dir, 
                                    output_dir=self.optflow_dir)
            self.hyoidTrackingWorker = HyoidBoneLocalizer(
                                    video_dir=self.video_dir, 
                                    hyoid_output_dir=self.hyoid_data_dir,
                                    c2_c4_output_dir=self.c2_c4_data_dir)
            self.DynamicClassifierWorker = DynamicFramesClassifier(
                                    video_data_dir=self.video_data_dir,
                                    video_name=self.video_name.lower(),
                                    video_size = self.capture.totalFrameCount,
                                    output_dir=[ self.dynamic_and_static_data_dir,
                                                self.dynamic_and_static_data_dir_after_filteration,
                                                self.dynamic_and_static_data_dir_ground_truth])
            self.PharyngealClassifierWorker = PharyngealClassifier(
                                    video_data_dir=self.video_data_dir,
                                    video_name=self.video_name.lower(),
                                    video_size = self.capture.totalFrameCount,
                                    dynamic_and_static_data_dir=self.dynamic_and_static_data_dir_after_filteration,
                                    output_dir=[ self.pharyngeal_data_dir,
                                                self.pharyngeal_data_dir_after_filteration,
                                                self.pharyngeal_data_dir_ground_truth])
            self.bolusSegmentationWorker = BolusSegmentor(
                                    video_dir=self.video_dir,
                                    output_dir=self.bolus_data_dir,
                                    pharyngial_data_dir=self.pharyngeal_data_dir_ground_truth)

            # Create a thread for each worker to run
            if not Path(self.frames_dir).exists():
                mkdir_ifnotexists(self.frames_dir)
                self.createThread(self.framesExtractorWorker, threadName= 'frames extraction')

            if not Path(self.optflow_dir).exists():
                mkdir_ifnotexists(self.optflow_dir)
                self.createThread(self.opticalFlowWorker, threadName= 'optical flow computation',
                                finish_callback_function=self.start_static_and_dynamic_classification_thread)
            else:
                if not Path(self.dynamic_and_static_data_dir).exists():
                    self.start_static_and_dynamic_classification_thread()
                else:
                    self.setupStaticDynamicDistribution()
                    if not Path(self.pharyngeal_data_dir).exists():
                        self.start_pharyngeal_classification_thread()
                    else:
                        self.setupPharyngealDistribution()
                        if not Path(self.bolus_data_dir).exists():
                            self.start_bolus_segmentation_thread()
                        else:
                            self.load_bolus_data()

            if not Path(self.hyoid_data_dir).exists() or not Path(self.c2_c4_data_dir).exists():
                self.createThread(self.hyoidTrackingWorker, threadName= 'hyoid bone and c2-c4 detection',
                                finish_callback_function=self.load_hyoid_data,
                                finish_callback_function_2=self.load_c2_c4_data)
            else:
                self.load_hyoid_data()
                self.load_c2_c4_data()

    def hyoidTrackingCheckBox(self, state):
        if state == QtCore.Qt.Checked:
            self.capture.hyoidbone_enable = True
        else:
            self.capture.hyoidbone_enable = False
        self.capture.displayCurrentFrame()

    def c2c4TrackingCheckBox(self, state):
        if state == QtCore.Qt.Checked:
            self.capture.c2_c4_enable = True
        else:
            self.capture.c2_c4_enable = False
        self.capture.displayCurrentFrame()

    def bolusCheckBox(self, state):
        if state == QtCore.Qt.Checked:
            self.capture.bolus_mask_enable = True
        else:
            self.capture.bolus_mask_enable = False
        self.capture.displayCurrentFrame()

    def load_hyoid_data(self):
        self.capture.hyoid_data_dir = pd.read_csv(self.hyoid_data_dir)
        self.capture.hyoid_data_available = True
        self.capture.displayCurrentFrame()

    def load_c2_c4_data(self):
        self.capture.c2_c4_data_dir = pd.read_csv(self.c2_c4_data_dir)
        self.capture.c2_c4_data_available = True
        self.capture.displayCurrentFrame()

    def load_bolus_data(self):
        self.capture.bolus_data_dir = self.bolus_data_dir
        self.capture.bolus_data_available = True
        self.capture.displayCurrentFrame()

    def closeEvent(self, event):
        if self.threadsNum:
            choice = QtWidgets.QMessageBox.question(self, 'Message','Process still running, Do you really want to exit?',QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if choice == QtWidgets.QMessageBox.Yes:
                for threadName, thread in self.threads.items():
                    if thread:
                        thread.quit()
                        thread.deleteLater()
                event.accept()
            else:
                event.ignore()

# function for launching a QApplication and running the ui and main window
def window():
    app = QApplication(sys.argv)
    win = MainWindow()
    stylesheet = """QScrollBar {
                    background-color: #FFFFFF;
                    }"""
    app.setStyleSheet(stylesheet)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    window()