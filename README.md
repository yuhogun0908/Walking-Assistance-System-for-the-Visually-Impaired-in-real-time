# WSVI
WASVI : Walking Assistance System for the Visually Impaired in real-time

## Methods
[1]. Object Detection
[2]. Multi Object Tracking
[3]. Traffic Light Detection
[4]. Pedestrian Intention Prediction

## Asuumption
The visually impaired has ability to follow the direction well with his cane on the sidewalk.
They don't visit the crowded place.
They go to fiamilar, and frequently visited place.

## Our Contribution
Make the system possible to predict the surrounding pedestrian movement.
Detect the traffic light and consider the signal which is dependent on the country
Detect the crosswalk and its direction(not yet)


- Overall Architecture
<img src="https://user-images.githubusercontent.com/67786803/170383356-6c5d97a2-ebcf-49f3-be70-e4ea23de6bf6.png" width="800" height="372">

- Detect Traffic Light Signal
<img src="https://user-images.githubusercontent.com/67786803/170383396-3cb3c162-3ef6-414d-9406-079654ca4bef.png" width="800" height="372">

- Detect Pedestrian Intention Prediction
<img src="https://user-images.githubusercontent.com/67786803/170383427-c5092c63-59cc-451f-96ae-b9f8309e9843.png" width="800" height="372">

- Detecting CrossWalk
<img src="https://user-images.githubusercontent.com/67786803/170383479-73d8a354-9ea1-452a-9194-073094ce039d.png" width="800" height="372">

## setup python packages environments
```
pip install -r requirements.txt
```

## DOWNLOAD OBJECT DETECTION MODEL
Download the yolor_p6.pt and save it in the yolormodel folder.
```
https://drive.google.com/file/d/10ZobLxTBr5r5gDZ1tq3uYn6Sq1YpoU0T/view?usp=sharing
<Reference : https://github.com/WongKinYiu/yolor>
```

## Run
```
python superman.py --video2frames True --videoName Final1.mp4 --use_cuda 1 
```

## Result 
- Detect the signal from the traffic light
<img src="https://user-images.githubusercontent.com/67786803/170382795-edd53e4f-5422-4981-a2c8-e054f4176037.png" width="800" height="372">

- Predict the pedestrian intention and warning
<img src="https://user-images.githubusercontent.com/67786803/170382750-19c7a66a-815c-4b16-8450-f0fdf3c09beb.png" width="800" height="372">

- Detect the crosswalk
<img src="https://user-images.githubusercontent.com/67786803/170382677-4c136deb-05eb-4370-8665-72423e85fe68.png" width="400" height="160">

## Time Break-down
<img src="https://user-images.githubusercontent.com/67786803/170382895-92fa8c1d-983c-4757-b5d9-3fdcd91502ae.png" width="800" height="372">

## Reference
- [1]. Wang, Chien-Yao, I-Hau Yeh, and Hong-Yuan Mark Liao. "You only learn one representation: Unified network for multiple tasks." arXiv preprint arXiv:2105.04206 (2021).
- [2]. Bouhsain, Smail Ait, Saeed Saadatnejad, and Alexandre Alahi. "Pedestrian intention prediction: A multi-task perspective." arXiv preprint arXiv:2010.10270 (2020).
- [3]. Romic, Kreimir, et al. "Real-time multiresolution crosswalk detection with walk light recognition for the blind." Advances in Electrical and Computer Engineering 18.1 (2018): 11-20.
