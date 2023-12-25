# Graduation_project

## Project Title 
AI-Powered Toolkit for Automated Swallowing Kinematic Analysis in X-Ray
Videofluoroscopy

## Problem summary
Swallowing is a well-coordinated neuromuscular process essential for human survival in which the food is transferred from the oral cavity into the stomach. Swallowing involves the coordination between multiple anatomical structures in the head/neck regions to function properly and many of these anatomical structures are shared pathways between swallowing and respiration [1]. 


https://user-images.githubusercontent.com/61396368/193665945-886038f6-24b7-4e7a-9cf5-b5a1674a8d70.mp4



As a result, any dysfunction or discoordination in the swallowing process can result in rerouting food from its original path to the stomach into the airway. Swallowing dysfunction is known as dysphagia and can lead to severe complications including malnutrition, dehydration, and aspiration pneumonia. 
![image](https://user-images.githubusercontent.com/61396368/193666088-9de11176-6462-4afd-90e5-9cda02eff94e.png)


Videofluroscopic swallowing study (VFSS) is considered the gold standard examination to diagnose dysphagia. Clinicians examine x-ray-recorded videos to evaluate all the aspects of the swallowing process including the bolus motion, airway protection, and hyoid bone motion [2]. The evaluation of VFSS is done in a frame-by-frame fashion which is subjective and requires trained clinicians to perform. Moreover, VFSS is a labor-intensive process in which clinicians track anatomical components that are small and hard to be visually detected in a series of x-ray images [3].


https://user-images.githubusercontent.com/61396368/193666966-e1683a8b-090b-400a-82d3-9e7d9909fe09.mp4



## Methodology
![image](https://user-images.githubusercontent.com/61396368/193666853-c69744e0-78a8-46c9-b947-c4a4d01ed9fd.png)

➢ First, we extract the swallowing segments within the VFSS videos through the calculation of optical flow which estimates the relative motion between each two subsequent frames. Optical flow between each consecutive VFSS frames is then introduced to a DNN that identifies frames that include bolus motion.A subsequent DNN is then used to separate the frames with motion that belong to the pharyngeal phase only.

➢ Second, we use a third deep neural network to automatically detect the location of hyoid bone with reference to C2 and C4 vertebras as its motion is highly associated with the level of airway protection and swallowing integrity.

➢ Third, we segment the bolus using another DNN as it is important to detect the airway invasion.

➢ Fourth, A graphical user interface will combine all these steps into a single application that takes an x-ray video with multiple swallows from a patient and produces a report about the status of the swallowing process in terms of normality given the results from the previously mentioned algorithms and comparison with normal/patient cohorts.

## Achievements and Skills Gained
1. Optical Flow (TVL1 – FlowNet2.0).
2. Deep Learning Frameworks (TensorFlow, Pytorch).
3. Deep Learning Architectures (U-Net, SSD, VGG16, Shallow Networks).
4. GPU Acceleration (CUDA), Multi-Processing, Multi-Threading.
5. Graphical User Interface (PyQt5).

## Main Results 

https://user-images.githubusercontent.com/61396368/193660517-9610bd03-dcd3-43c1-8316-57cb02707b18.mp4


## Discussion And Conclusion
This cascaded pipeline can support physicians while making diagnostic decisions. The dynamic frames model reduces the initial examination time by more than 50\%. Pharyngeal model further separate pharyngeal frames by only working on the dynamic frames which will reduce the overall inference time of subsequent networks. The system can also provide a joint model which localize the third and fourth cervical vertebral (C3 and C4) anatomic scalars along with the hyoid bone with a mean average precision more than 71% at IoU=0.5 which add interpretive value to the judgement of hyoid displacement [4]. The segmentation model can extract the food bolus with 78% Jaccard Index which provide objective method for clinicians to track bolus flow or detect the bolus residue during VFSS.

## References 
1. Clave, P., & Shaker, R. (2015). Dysphagia: current reality and scope of the problem. Nature reviews. Gastroenterology & hepatology, 12(5), 259–270.
2. Palmer, J. B., Kuhlemeier, K. V., Tippett, D. C., & Lynch, C. (1993). A protocol for the videofluorographic swallowing study. Dysphagia, 8(3), 209–214.
3. McCullough, G. H., Wertz, R. T., Rosenbek, J. C., Mills, R. H., Webb, W. G., & Ross, K. B. (2001). Inter- and intrajudge reliability for videofluoroscopic swallowing evaluation measures. Dysphagia, 16(2), 110–118.
4. Zhang, Z., Coyle, J. L., & Sejdi ́c, E. (2018). Automatic hyoid bone detection in fluoroscopic images using deep learning. Scientific reports, 8(1), 12310.

## Future Work Suggestion
Making a subsequent model to automatically detect the bolus residue and the airway invasion (food presence in larynx). Detecting pharyngeal phase based on dynamic region of interest (ROI) from the C2-C4 Segment.

## Group Picture
<img src="imgs/group_pics/pic.jpg">
