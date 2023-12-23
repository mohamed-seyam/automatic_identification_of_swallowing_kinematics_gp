from PyQt5.QtCore import QObject
from helpers.helpers.convert import convert_video_to_frames

class FramesExtractorGUI(QObject):
    def __init__(self, video_dir, frames_dir):
        super().__init__()
        self.video_dir = video_dir
        self.frames_dir = frames_dir

    def run(self):
        convert_video_to_frames(self.video_dir, self.frames_dir)
