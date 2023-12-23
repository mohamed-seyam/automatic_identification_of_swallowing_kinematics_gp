from typing import Any
import tensorflow as tf
import numpy as np
import pandas as pd 
import os 
from itertools import cycle
from PyQt5.QtCore import QObject

class DynamicFrameClassifier:
    def __init__(self, video_dir = None, subject_name = None, output_dir = None, weights_dir = None):
        self.video_dir = video_dir
        self.subject_name = subject_name,
        self.output_dir = output_dir
        self.weights_dir = weights_dir

        self.model = None 

    def _create_model(self, ):
        model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(16, (3,3) ,activation = 'relu' , input_shape = (128 , 128 , 1) ) , 
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Conv2D(64,(3,2), activation = 'relu'),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64,activation = 'relu'),
                tf.keras.layers.Dense(1,activation = "sigmoid")
            ])

        model.compile(loss = "binary_crossentropy", optimizer = tf.keras.optimizers.RMSprop(learning_rate = .001),
                        metrics = ['accuracy'])

        self.model = model

    def _load_data_in_generator(self, ):
        image_generator = tf.keras.preprocessing.image.ImageDataGenerator(samplewise_center=True, 
                                                                          samplewise_std_normalization= True)
        
        generator = image_generator.flow_from_directory(
                            directory = self.frames_dir,
                            target_size=(128, 128),
                            color_mode="grayscale",
                            classes = ["optflow"],
                            class_mode=None,
                            shuffle=False,
                            )
        
        return generator
    
    def _get_predictions(self, generator):
        predictions = self.model.predict(generator).reshape(-1)
        for i in range(len(predictions)):
            predictions[i] = 1 if predictions[i] >= 0.7 else 0

        return predictions

    def _save_dynamic_regions(self, begin_arr, end_arr):
        dynamic_regions_df = pd.DataFrame({'begin' : begin_arr, 'end': end_arr})
        dynamic_regions_df.to_csv(self.output_dir)

    def _remove_notches(self, begin_arr, end_arr, static_notch_thr, dynamic_notch_thr):
        df = pd.DataFrame({'begin' : begin_arr, 'end': end_arr})
        df['frames_between'] = np.abs(df['end'].sub(df['begin']))
        df = df.loc[df['frames_between'] > dynamic_notch_thr, ['begin', 'end']]
        df['begin_shifted'] = df.begin.shift(periods=-1, axis='index')
        df.fillna(200000, inplace=True)
        df['frames_ahead'] = np.abs(df['end'].sub(df['begin_shifted']))
        temp_df = df[['end', 'frames_ahead']].copy()
        temp_df.where(temp_df['frames_ahead'] > static_notch_thr, np.nan, inplace=True)
        temp_df.fillna(method='bfill', inplace=True)
        df['end'] = temp_df['end']
        df.drop(columns=['frames_ahead'], inplace=True)
        df.drop_duplicates(subset='end', keep='first', inplace=True)
        df = df.astype('int')
        return df['begin'].values, df['end'].values
    

    def _filter_predictions(self, predictions, window_size, thr):
        filter_window = sum([e for e in predictions[:window_size]])
        for idx in range(len(predictions)):
            if idx < window_size:
                pass
            filter_window = filter_window + predictions[idx] - predictions[idx-window_size]
            if filter_window > thr:
                predictions[idx-window_size+1: idx + 1] = 1
                idx = idx + window_size - thr
        return predictions
    
    def run(self):
        self._create_model()
        self.model.load_weights('./static_dynamic_model/cp-0005.ckpt')
        print(self.model.summary())

        generator = self._load_data_in_generator()
        predictions = self._get_predictions(generator)

        dynamic_frames_begin_indices, dynamic_frames_end_indices = self._get_dynamic_frames_indices(predictions)
        # save the dynamic frames indices before filtration
        #TODO define the save path
        self._save_dynamic_regions(dynamic_frames_begin_indices, dynamic_frames_end_indices)
        # filter the predictions to remove the noise
        predictions_f = self._filter_predictions(predictions, window_size=20, thr=10)
        dynamic_frames_begin_indices_f, dynamic_frames_end_indices_f = self._get_dynamic_frames_indices(predictions_f)
        dynamic_frames_begin_indices_f, dynamic_frames_end_indices_f = self._remove_notches(dynamic_frames_begin_indices_f, 
                                                                                            dynamic_frames_end_indices_f, 
                                                                                            static_notch_thr=5, 
                                                                                            dynamic_notch_thr=1)
        # save the dynamic frames indices after filtration
        #TODO define the save path 
        self._save_dynamic_regions(dynamic_frames_begin_indices_f, dynamic_frames_end_indices_f)

        # Get the ground Truth 
        gt_labels = self._find_gt_dynamic_frames(self.video_dir, self.subject_name)
        dynamic_frames_begin_indices_gt, dynamic_frames_end_indices_gt = self._get_dynamic_frames_indices(gt_labels)
        #TODO define the save path
        self._save_dynamic_regions(dynamic_frames_begin_indices_gt, dynamic_frames_end_indices_gt)


    def _get_dynamic_frames_indices(self, arr):
        dynamic_frames_begin_indices, dynamic_frames_end_indices = [], []
        
        for idx in range(1, len(arr)):
            prev = arr[idx-1]
            curr = arr[idx]

            if prev == 0 and curr == 1:
                dynamic_frames_begin_indices.append(idx)
            
            if prev == 1 and curr == 0 and len(dynamic_frames_begin_indices) != 0:
                dynamic_frames_end_indices.append(idx)
            
        if len(dynamic_frames_begin_indices) != len(dynamic_frames_end_indices):
            dynamic_frames_end_indices.append(len(arr)-1)

        return dynamic_frames_begin_indices, dynamic_frames_end_indices
        

    def _find_gt_dynamic_frames(subjects_dir, subject_name):  
        folder_name = "frames"      
        dynamic_df_gt = pd.read_csv("./data/v-stops)_all.csv")
        participant_name = subject_name[0:-1].upper()
        participant_num = subject_name[-1].lower()

        non_pharyngeal_list = []

        data = dynamic_df_gt.loc[(dynamic_df_gt['Participant'] == participant_name) & (dynamic_df_gt['file_num'] == participant_num)]
        for i in data.index:
            non_pharyngeal_parts = [f'{j}.jpeg'  for j in range(int(data["start"][i]), int(data["end"][i])+1)]
            non_pharyngeal_list.append(non_pharyngeal_parts)

        subject_pharyngeal = [item for sublist in non_pharyngeal_list for item in sublist]

        subject_non_pharyngeal = sorted(list(set(os.listdir(os.path.join(subjects_dir, folder_name))) - set(subject_pharyngeal)))
        _temp = sorted([int(item.split('.')[0]) for item in subject_non_pharyngeal])
        subject_non_pharyngeal = [f"{item}.jpeg" for item in _temp]

        zip0 = list(zip(subject_pharyngeal, cycle('0'))) 
        zip1 = list(zip(subject_non_pharyngeal, cycle('1')))
        
        zipped = zip0 + zip1
        sorted_zipped = sorted(zipped, key = lambda x: int(x[0].split('.')[0]))

        subject_label = []     # list of zeros and ones [0, 0, 0, 0, 1, 1, 1, 1, ...]

        for i in range(len(sorted_zipped)):
            subject_label.append(int(sorted_zipped[i][1]))

        return subject_label[:-1]


               
class DynamicFrameClassifierGui(QObject):
    def __init__(self, algorithm: DynamicFrameClassifier):
        super().__init__()
        self.algorithm = algorithm
    
    def set_video_dir(self, video_dir):
        self.algorithm.video_dir = video_dir
    
    def set_output_dir(self, output_dir):
        self.algorithm.output_dir = output_dir

    def set_subject_name(self, subject_name):
        self.algorithm.subject_name = subject_name

    def run(self):
        self.algorithm.run()


