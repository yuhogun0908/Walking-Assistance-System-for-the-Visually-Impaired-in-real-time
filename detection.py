from platform import java_ver
from this import d
from yolormodel.utils.plots import plot_one_box
from yolormodel.models.models import *
from yolormodel.utils.datasets import LoadImages
from yolormodel.utils.torch_utils import time_synchronized
from yolormodel.utils.general import ( check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)

import torch
import time
from pathlib import Path
from utils_general import *
import cv2


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def object_detect(args, net):

    # dp = post.DmPost(opt)
    cfg = args.cfg
    device = args.device
    weights = args.weights
    imgsz = args.img_size
    names = args.names
    iou_thres = args.iou_thres
    classes = args.classes
    agnostic_nms = args.agnostic_nms
    augment = args.augment



    dataPath = args.videoName
    images=os.listdir(os.path.join('videos',dataPath.split('.')[0]))
    images_len = len(images)
    format = images[0].split('.')[-1]
    # sort images
    images_sort = []
    for idx in range(images_len):
        images_sort.append('{}/{}.{}'.format(os.path.join('videos',dataPath.split('.')[0]),idx,format))
    source = images_sort

    vid_cap = cv2.VideoCapture(os.path.join('videos',dataPath))
    

    savePath = os.path.join('result',dataPath.split('.')[0])
    savePathImage = os.path.join(savePath,'frames')
    savePathVideo = os.path.join(savePath,'video')
    Path(savePathImage).mkdir(exist_ok=True, parents=True)
    Path(savePathVideo).mkdir(exist_ok=True, parents=True)


    model = Darknet(cfg,imgsz).cuda()
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
    model.to(device).eval()

    half = 'cuda' in device
    if half:
        model.half()

    dataset = LoadImages(source, img_size=imgsz, auto_size=32)

    names = load_classes(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    intent_color = [[0,255,0], [0,0,255]]
    traffic_colors = [[0,0,0],[0,0,255],[255,225,255],[0,255,0]]

    print(len(colors))
    to = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img

    # images_sort = ['videos/test3/115.png']
    frame = 0
    frame_info = {}
    traffic_dict = {}
    output_dict = {}
    max_num = 0

    traffic_frame = 0
    traffic_light = {}

    pede_predict = {}
    vid_writer = None
    fourcc = 'mp4v'  # output video codec
    if isinstance(vid_writer, cv2.VideoWriter):
        vid_writer.release()  # release previous video writer
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_writer = cv2.VideoWriter(savePathVideo+'/video.mp4', cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))

    red_flag = False
    red_frame = 0

    _ = model(img.half() if half else img) if 'cpu' not in device else None  # run once
   
    for t_idx, data in enumerate(dataset):
        print("-"*100)
        print("frame number (t_idx) : ", t_idx)

        start = time.time()

        path, img, im0s, _ ,recover= data
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()

        with torch.no_grad():
            pred = model(img, augment=augment)[0]
             
        # print('before',pred, '\n')
        # Apply NMS
        pred = non_max_suppression(pred,  args.conf_thres, iou_thres, classes=classes, agnostic= agnostic_nms)
        # pdb.set_trace()
        # # pred_cross = non_max_suppression(pred_cross,  opt.conf_thres, opt.iou_thres, classes=0, agnostic= opt.agnostic_nms)
        
        # print(conf, iou_thres,classes, agnostic_nms)

        end = time.time()

        print("object detection time : ", f"{end - start:.5f} sec")
        # print('after',pred,'\n')
        t2 = time_synchronized()
        # Process detections

        for i, det in enumerate(pred):  # detections per image
            # print('pred' , len(pred))
            p, s, im0 = path, '', im0s
            
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                count = 0
                t_count = 0
                for idx, (*xyxy, conf, cls) in enumerate(det):
                    sxywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    label = '%s %.2f' % (names[int(cls)], conf)
                    # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                    ######
                    #print(names[int(cls)], xyxy)
                    
                    #if names[int(cls)] == 'person':
                        #print("test : ", idx)
                        #print(xyxy)

                    if names[int(cls)] == 'person':
                        #print("person")
                        count += 1
                        if count == 1:
                            bb_list=[]
                        x1,y1,x2,y2 = xyxy
                        x1 = x1.detach().cpu().numpy()
                        y1 = y1.detach().cpu().numpy()
                        x2 = x2.detach().cpu().numpy()
                        y2 = y2.detach().cpu().numpy()
                        
                        w = int(x2-x1)
                        h = int(y2-y1)
                        x = int(x1)
                        y = int(y1)
                        bb_list.append([x,y,w,h])

                    if names[int(cls)] == 'traffic light':
                        #print("traffic")
                        t_count += 1
                        if t_count == 1:
                            tt_list=[]
                        x1,y1,x2,y2 = xyxy
                        x1 = x1.detach().cpu().numpy()
                        y1 = y1.detach().cpu().numpy()
                        x2 = x2.detach().cpu().numpy()
                        y2 = y2.detach().cpu().numpy()
                        
                        w = int(x2-x1)
                        h = int(y2-y1)
                        x = int(x1)
                        y = int(y1)
                        tt_list.append([x,y,w,h])

                try:
                    frame_info.update({frame:bb_list})
                    #print(bb_list)
                except:
                    pass

                # try:
                #     traffic_info.update({traffic_frame:tt_list})
                #     #print(bb_list)
                # except:
                #     pass

                
            # save Image        
            # cv2.imwrite(os.path.join(savePathImage,'{}.{}'.format(frame,format)), im0)
            # cv2.imwrite(os.path.join(savePathImage,'{}.{}'.format(frame_idx,format)), im0)
            # cv2.imwrite('result1.jpg', im0)
            
            # save Video 

            # fps = vid_cap.get(cv2.CAP_PROP_FPS)
            # w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # h = int(vid_cap.gest(cv2.CAP_PROP_FRAME_HEIGHT))
            # vid_writer.write(im0)

        
        
        ### traffic light object tracking
        dist_threshold = 80
        
        #print(str(traffic_frame) + " traffic : ", tt_list)
        
        start = time.time()

        if traffic_frame == 0:
            traffic_list = []
            traffic_num = len(tt_list)

            for i in range(traffic_num):
                traffic_obj = {i:tt_list[i]}
                traffic_list.append(traffic_obj)
            
            traffic_dict[str(traffic_frame)] = traffic_list
            #print(output_dict)
            max_num_traffic = traffic_num

        else:
            prev_traffic_list = traffic_dict[str(int(traffic_frame-1))]
            cur_traffic_list = tt_list

            max_num_traffic, traffic_list = traffic_light_tracking(max_num_traffic, prev_traffic_list, cur_traffic_list, dist_threshold)
            traffic_dict[str(traffic_frame)] = traffic_list
        
        # "person" object tracking    
        if frame == 0:
            #print("herehere")
            object_list = []
            num_obj = len(bb_list)

            for i in range(num_obj):
                # obj = {'id':i, 'cur_box':bb_list[i]}
                obj = {i:bb_list[i]}
                object_list.append(obj) # [ {id:[]}, {id:[]}, {id:[]}, ... ]
            
            output_dict[str(frame)] = object_list
            #print(output_dict)
            max_num = num_obj
        else:
            # if frame >= 280: # 280, 336
            #     continue
            if frame < 280:
                if len(bb_list) > 2:
                    object_list = output_dict[str(int(frame)-1)]
                    output_dict[str(frame)] = object_list
                    #frame += 1
                else:
                    #print("here : ", output_dict.keys())
                    prev_obj_list = output_dict[str(int(frame)-1)]
                    #print("prev : ", prev_obj_list)
                    cur_obj_list = bb_list
                    #print("cur : ", cur_obj_list)

                    max_num, object_list = object_tracking(max_num, prev_obj_list, cur_obj_list)
                    output_dict[str(frame)] = object_list
            else:
                output_dict[str(frame)] = []
                
        ################ person end ####################    
        
        end = time.time()
        print("object tracking time : ", f"{end - start:.5f} sec")

        # inference for petestrian prediction
        if frame >= 16 and frame % 8 == 0:
            start = time.time()
            pede_predict = inference_(frame, output_dict, net)
            end = time.time()
            print("Intention prediction time : ", f"{end - start:.5f} sec")

        print("person track : ", output_dict[str(frame)])
        if len(pede_predict) != 0:
            print("person move intention : ", pede_predict)
        print("traffic light : ", traffic_dict[str(traffic_frame)])
        


        # Draw move intention
        for ped_id in pede_predict.keys():
            cur_bb = []
            for obj in output_dict[str(frame)]:
                id, bb = list(obj.items())[0]
                if ped_id == id:
                    cur_bb = [bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3]]
                    break
            pede_intent = pede_predict[ped_id]

            if len(cur_bb) != 0:
                if pede_intent == 1:
                    print("warning : ", cur_bb)
                    plot_one_box(cur_bb, im0, label='warning', color=intent_color[1], line_thickness=3)
                else:
                    print("safe : ", cur_bb)
                    plot_one_box(cur_bb, im0, label='safe', color=intent_color[0], line_thickness=3)


        # Draw Traffic Light
        start = time.time()

        for traffic_obj in traffic_dict[str(traffic_frame)]:
            traffic_id, traffic_bb = list(traffic_obj.items())[0]
            x,y,w,h = traffic_bb
            
            crop = get_crop_img(im0, [x,y,x+w,y+h])

            color_level, cnt = detect_color(crop)
            print('color level level',color_level)
            print('red number', cnt['red'])
            if color_level['white'] > 5:
                label = '%s' % ('car traffic light' )
                traffic_cls = 3
            
            else:
                if color_level['green'] > color_level['red'] and color_level['green']  > color_level['white'] :
                    label = '%s %.4f' % ('car traffic light', color_level['green'] )
                    traffic_cls = 3
                elif color_level['white']  > color_level['red'] and color_level['white']  > color_level['green'] :
                    #label = '%s %.4f' % ('white', color_level['white'] )
                    label = '%s %.4f' % ('CROSS', color_level['white'] )
                    traffic_cls = 2
                elif color_level['red'] > color_level['white'] and   color_level['red'] > color_level['green']:
                    #label = '%s %.4f' % ('red', color_level['red'] )
                    traffic_cls = 1
                    label = '%s %.4f' % ('STOP', color_level['red'] )

                    if red_flag == False and t_idx > 10 and traffic_id == 1:
                        red_frame = t_idx + 120
                        label = '%s %.4f' % ('HURRY UP', color_level['red'] )
                        red_flag = True
                    elif t_idx < red_frame and red_flag == True and traffic_id == 1:
                        label = '%s %.4f' % ('HURRY UP', color_level['red'] )
                        
                    if t_idx == red_frame and red_flag == True and traffic_id == 1:
                        red_flag = False
                        label = '%s %.4f' % ('STOP', color_level['red'] )
                        red_frame = 0
                    

                else:
                    label = '%s' % ('black' )
                    traffic_cls = 0
            #print("test : ", label)

            plot_one_box([x,y,x+w,y+h], im0, label=label, color=traffic_colors[int(traffic_cls)], line_thickness=3)

        end = time.time()
        print("traffic light time : ", f"{end - start:.5f} sec")
            

        traffic_frame += 1
        frame += 1

        cv2.imwrite(os.path.join(savePathImage,'{}.{}'.format(t_idx,'jpg')), im0)
        vid_writer.write(im0)

