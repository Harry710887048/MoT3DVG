# MoT3DVG: A Benchmark for Outdoor 3D Visual Grounding with Motion-Aware Descriptions and Temporal Cues
## About Dataset
[MoT3DVG](https://www.kaggle.com/datasets/nuonepeacey/mot3dvg/data) is built upon the nuScenes dataset, with additional motion-aware descriptions supplemented to provide language prompts for outdoor 3D visual grounding task.
<img width="762" height="238" alt="image" src="https://github.com/user-attachments/assets/f1a8f1ed-1409-4034-aa29-cf5198187a10" />

Please download the official nuScenes 3D object detection dataset and organize the download files as follows：

├── data
│   ├── nuscenes
│   │   │── v1.0-trainval
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── v1.0-trainval  
In MoT3DVG, we integrate bounding box annotations, language prompts, and other relevant information into a unified PKL file. Please put this PKL file under the directory /nuscenes/v1.0-trainval.
<img width="1328" height="401" alt="image" src="https://github.com/user-attachments/assets/7e0d432c-e2b9-4fbf-8503-f57114a907f1" />
