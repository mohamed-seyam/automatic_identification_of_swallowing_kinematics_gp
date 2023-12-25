import os

import numpy as np 

from pathlib import Path
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QImage, qRgb

def convert_video_to_frames(video_file_path, output_directory):
    video_name = Path(video_file_path).stem
    cmd = "ffmpeg -i %s -start_number 0 -vsync 0 %s/%s_%%05d.png" % (video_file_path, output_directory, video_name)
    os.system(cmd)

def convert_img_to_pyqt_image(image, copy=False):
    gray_color_table = [qRgb(i, i, i) for i in range(256)]
    if image is None:
        return QImage()

    image = np.require(image, np.uint8, 'C')
    if image.dtype == np.uint8:
        if len(image.shape) == 2:
            qim = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_Indexed8)
            qim.setColorTable(gray_color_table)
            return qim.copy() if copy else qim

        elif len(image.shape) == 3:
            if image.shape[2] == 3:
                qim = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
                return qim.copy() if copy else qim
            elif image.shape[2] == 4:
                qim = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_ARGB32)
                return qim.copy() if copy else qim
            

def convert_cv2_to_pixel_map(cv2_image):
    qimg = convert_img_to_pyqt_image(cv2_image, copy = True)
    return QPixmap(qimg)



