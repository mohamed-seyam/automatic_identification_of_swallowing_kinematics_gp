import os 
import cv2
import tensorflow as tf 
import pandas as pd 
import numpy as np 


class BolusSegmenter:
    def __init__(self, video_dir, output_dir, pharyngeal_data_dir):
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.pharyngeal_data_dir = pharyngeal_data_dir
        self.checkpoint_path = None 
        self.cap = None
        self.model = None

    def set_checkpoint_path(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path

    def _create_model(self, input_shape, n_classes = 1):
        inputs = tf.keras.layers.Input(input_shape)

        s1, p1 = self._encoder_block(inputs, 64)
        s2, p2 = self._encoder_block(p1, 128)
        s3, p3 = self._encoder_block(p2, 256)
        s4, p4 = self._encoder_block(p3, 512)

        b1 = self._conv_block(p4, 1024) 

        d1 = self._decoder_block(b1, s4, 512)
        d2 = self._decoder_block(d1, s3, 256)
        d3 = self._decoder_block(d2, s2, 128)
        d4 = self._decoder_block(d3, s1, 64)

        if n_classes == 1:
            activation = 'sigmoid'
        else:
            activation = 'softmax'

        outputs = tf.keras.layers.Conv2D(n_classes, 1, padding="same", activation=activation)(d4)  #Change the activation based on n_classes

        model = tf.keras.models.Model(inputs, outputs, name="U-Net")
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _conv_block(self, input, num_filters):
        x = tf.keras.layers.Conv2D(num_filters, 3, padding = "same")(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.Conv2D(num_filters, 3, padding = "same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        return x 
    
    def _encoder_block(self, input, num_filters):
        x = self._conv_block(input, num_filters)
        p = tf.keras.layers.MaxPool2D((2,2))(x)
        return x, p  
    
    def _decoder_block(self, input, skip_features, num_filters):
        x = tf.keras.layers.Conv2DTranspose(num_filters, (2,2), strides = 2, padding = "same")(input)
        x = tf.keras.layers.Concatenate()[x, skip_features]
        x = self._conv_block(x, num_filters)
        return x 
        
    def _open_video_capture(self,):
        self.cap = cv2.VideoCapture(self.video_dir)

    def _load_target_regions_indices(self, ):
        target_regions = pd.read_csv(self.pharyngeal_data_dir)
        target_indices = []
        for region in target_regions.values:
            target_indices += list(range(region[1], region[2] + 1))
    
    def _process_frame(self, frame, frame_idx, first_frame = False):
        frame = cv2.resize(frame, (512, 512))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame /= 255.0
        frame = np.expand_dims(frame, axis = 2)
        frame = np.expand_dims(frame, axis = 0)
        frame_tensor = tf.convert_to_tensor(frame, dtype = tf.float32)
        return frame_tensor
    
    def _run_inference(self, frame_tensor):
        output = (self.model.predict(frame_tensor)[0,:,:,0] > 0.5).astype(np.uint8) * 255
        return output

    def _save_output_as_img(self, output, target_regions_indices, frame_idx):
        cv2.imwrite(os.path.join(self.output_dir, f"{target_regions_indices[frame_idx]}.png"), output)
    

    def run(self):
        self.model = self._create_model((512, 512, 1))
        self.model.load_weights(self.checkpoint_path)
        self._open_video_capture()

        target_regions_indices = self._load_target_regions_indices()
        curr_target_idx = 0
        # read first frame 
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_regions_indices[curr_target_idx])
        ret, frame = self.cap.read()
        curr_target_idx += 1

        while ret:
            frame_tensor = self._process_frame(frame, curr_target_idx)
            output = self._run_inference(frame_tensor)
            self._save_output_as_img(output, target_regions_indices, curr_target_idx)
            # read the next frame 
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_regions_indices[curr_target_idx])
            ret, frame = self.cap.read()
            curr_target_idx += 1
            if curr_target_idx == len(target_regions_indices):
                break
        
        self.cap.release()

    


        

        