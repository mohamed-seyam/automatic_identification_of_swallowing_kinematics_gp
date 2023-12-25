import tensorflow as tf
from auto_identification_of_vfss.modules import DynamicFrameClassifier

class PharyngealClassifier:
    def __init__(self, video_dir, video_name, dynamic_static_frames_dir, output_dir):
        self.video_dir = video_dir
        self.video_name = video_name
        self.dynamic_static_frames_dir = dynamic_static_frames_dir
        self.output_dir = output_dir

        self.dynamic_frame_clf = DynamicFrameClassifier(self.video_dir, self.video_name, self.dynamic_static_frames_dir )
    
    def _create_model(self):
        pre_trained_model = tf.keras.applications.VGG16(input_shape=(224, 224, 3), include_top=False, weights=None)
        for layer in pre_trained_model.layers:
                layer.trainable = False
        
        last_layer = pre_trained_model.get_layer('block5_pool')
        last_output = last_layer.output
        
        x = tf.keras.layers.GlobalMaxPooling2D()(last_output)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(pre_trained_model.input, x)
               
        metrics = [
                tf.keras.metrics.TruePositives(name='tp'),
                tf.keras.metrics.FalsePositives(name='fp'),
                tf.keras.metrics.TrueNegatives(name='tn'),
                tf.keras.metrics.FalseNegatives(name='fn'), 
                tf.keras.metrics.SpecificityAtSensitivity(0.5),
                tf.keras.metrics.SensitivityAtSpecificity(.5),
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        ]

        model.compile(loss = "binary_crossentropy", optimizer = "adam",
                          metrics = metrics)

        self.model = model
    

    def run(self):
        self.dynamic_frame_clf.run()

        
         
         
class PharyngealClassifierGUI(QObject):
    def __init__(self, algorithm: PharyngealClassifier):
        super().__init__()
        self.algorithm = algorithm