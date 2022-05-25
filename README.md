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
<img src="https://user-images.githubusercontent.com/67786803/170381742-b07b64f3-0c89-45de-a773-feb5336a18f0.png" width="800" height="372">

- Detect Traffic Light Signal
<img src="https://user-images.githubusercontent.com/67786803/170381958-f0d7063d-7903-41a2-ae23-dd9dff69f3cc.png" width="800" height="372">

- Detect Pedestrian Intention Prediction
<img src="https://user-images.githubusercontent.com/67786803/170382197-5a300243-ac16-4462-b564-0aec67191c48.png" width="800" height="372">

-Detecting CrossWalk
<img src="https://user-images.githubusercontent.com/67786803/170382339-4d4ade60-d399-4e40-8703-fef035216c3a.png" width="800" height="372">

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

-Predict the pedestrian intention and warning
<img src="https://user-images.githubusercontent.com/67786803/170382750-19c7a66a-815c-4b16-8450-f0fdf3c09beb.png" width="800" height="372">

-Detect the crosswalk
<img src="https://user-images.githubusercontent.com/67786803/170382677-4c136deb-05eb-4370-8665-72423e85fe68.png" width="247" height="112">

## Time Break-down
<img src="https://user-images.githubusercontent.com/67786803/170382895-92fa8c1d-983c-4757-b5d9-3fdcd91502ae.png" width="800" height="372">

## Reference
https://arxiv.org/abs/2105.04206
https://arxiv.org/pdf/2010.10270.pdf

