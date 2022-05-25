# WSVI
WASVI : Walking Assistance System for the Visually Impaired in real-time

## Methods
[1]. Object Detection
[2]. Multi Object Tracking
[3]. Traffic Light Detection
[4]. Pedestrian Intention Prediction

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

## Reference
https://arxiv.org/abs/2105.04206
https://arxiv.org/pdf/2010.10270.pdf

