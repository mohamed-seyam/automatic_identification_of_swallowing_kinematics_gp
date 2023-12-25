import os
import cv2
import tensorflow as tf
import numpy as np 
import pandas as pd

from PyQt5.QtCore import QObject

from object_detection.builders import model_builder
from object_detection.utils import config_util

class HyoidBoneLocalizer:
    def __init__(self, video_dir, hyoid_output_dir, 
                 c2_c4_output_dir):
        self.video_dir = video_dir
        self.hyoid_output_dir = hyoid_output_dir
        self.c2_c4_output_dir = c2_c4_output_dir

        self.detection_model_path = os.path.join(os.getcwd(), 'My_ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8')
        self.detection_model_config_path = os.path.join(self.detection_model_path, 'pipeline.config')
        self.labels_map_file_path = os.path.join(self.detection_model_path, 'label_map.pbtxt')

        self.configs = None 
        self.detection_model = None
        self.ckpt = None


        self.hyoid_bone_df = pd.DataFrame(columns=['ymin', 'xmin', 'ymax', 'xmax'])
        self.c2_c4_df = pd.DataFrame(columns=['ymin', 'xmin', 'ymax', 'xmax'])
    

    
    def _load_configs(self):
        configs = config_util.get_configs_from_pipeline_file(self.detection_model_config_path)
        self.configs = configs
    
    def _load_model(self):
        self.detection_model = model_builder.build(model_config=self.configs['model'], is_training=False)
        self.ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)
        self.ckpt.restore(os.path.join(self.detection_model_path, 'ckpt-21')).expect_partial()

    def _open_video_capture(self):
        self.cap = cv2.VideoCapture(self.video_dir)

    def _preprocess_frame(self, frame):
        frame = np.expand_dims(frame, axis=0)
        frame_tensor = tf.convert_to_tensor(frame, dtype=tf.float32)
        return frame_tensor
    
    @tf.function
    def _run_inference(self, frame_tensor):
        image, shapes = self.detection_model.preprocess(frame_tensor)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        return detections

    def _save_data_frames_results(self):
        self.hyoid_bone_df.to_csv(self.hyoid_output_dir)
        self.c2_c4_df.to_csv(self.c2_c4_output_dir)
    
    def run(self):
        self._load_configs()
        self._load_model()
        self._open_video_capture()
        curr_frame_idx = 0
        ret, frame = self.cap.read()
        while ret:
            frame_tensor = self._preprocess_frame(frame)
            detections = self._run_inference(frame_tensor)
            
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
            
            # filter out non hyoid bone detections
            try:
                first_hyoid_bone_idx = list(detections['detection_classes']).index(0)
                box = detections['detection_boxes'][first_hyoid_bone_idx]
                self.hyoid_bone_df.loc[len(self.hyoid_bone_df.index)] = [box[0], box[1], box[2], box[3]]
            
            except:
                self.hyoid_bone_df.loc[len(self.hyoid_bone_df.index)] = [np.nan, np.nan, np.nan, np.nan]


            # filter out non c2-c4 detections
            try: 
                first_c2_c4_idx = list(detections['detection_classes']).index(1)
                box = detections['detection_boxes'][first_c2_c4_idx, :]
                self.c2_c4_df.loc[len(self.c2_c4_df.index)] = [box[0], box[1], box[2], box[3]]  
            except:
                self.c2_c4_df.loc[len(self.c2_c4_df.index)] = [np.nan, np.nan, np.nan, np.nan]

            # read the next frame 
            ret, frame = self.cap.read()
            curr_frame_idx += 1
        
        self.cap.release()
        self._save_data_frames_results()
        

class HyoidBoneLocalizerGUI(QObject):
        def __init__(self, algorithm: HyoidBoneLocalizer):
            super().__init__()
            self.algorithm = algorithm
        
        def set_video_dir(self, video_dir):
            self.algorithm.video_dir = video_dir
        
        def set_output_dir(self, output_dir):
            self.algorithm.output_dir = output_dir
        
        def run(self):
            self.algorithm.run()
        
