import cv2 as cv
import numpy as np
import os
from pathlib import Path
import pdb
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import cv2
np.random.seed(44)
def get_crop_img(img,bbox):
    """
    img : shape (w, h , ch)
    """
    y1,x1,y2,x2 = bbox
    print(img.shape)
    print(bbox)

    # crop = img[int(bbox[1]):int(bbox[1])+int(bbox[3]), int(bbox[0]):int(bbox[2]) :]
    crop = img[int(x1):int(x2), int(y1):int(y2), :]
    return crop

def video2frames(Rootdir,videoName,resize, size):
    rawVideo = cv.VideoCapture(os.path.join(Rootdir,videoName))
    n_frame = int(rawVideo.get(cv.CAP_PROP_FRAME_COUNT))
    # frames = np.empty((n_frame,), dtype=np.ndarray)

    savePath = os.path.join(Rootdir,videoName.split('.')[0])
    Path(savePath).mkdir(exist_ok=True, parents=True)
    for frame_idx in range(n_frame):
        _,frame =rawVideo.read()
        #print(frame.shape)
        if resize:
            frame = cv.resize(frame, dsize=size, interpolation=cv.INTER_LINEAR)
        cv.imwrite(os.path.join(savePath,'{}.png'.format(frame_idx)),frame)
        # PIL.Image.fromarray(video_transform.numpy().astype('uint8')).save(os.path.join(torchRootDir,Name.rstrip('.avi'),Name.replace('avi','pt')))
        # Image.fromarray(img).save(f, format='PNG')

    print('Video  {}'.format(videoName))
    print(' # framess {}'.format(frame_idx+1))

def get_distance(box_a, box_b): # [x, y, w, h]
    center_a_x = box_a[0] + box_a[2]/2
    center_a_y = box_a[1] + box_a[3]/2

    center_b_x = box_b[0] + box_b[2]/2
    center_b_y = box_b[1] + box_b[3]/2

    return math.sqrt((center_a_x - center_b_x)**2 + (center_a_y - center_b_y)**2)

def traffic_light_tracking(max_num, prev_obj_list, cur_obj_list, threshold): # threshold = 100

    object_list = []

    matrix = np.zeros((len(prev_obj_list), len(cur_obj_list)))
    for i in range(len(prev_obj_list)):
        for j in range(len(cur_obj_list)):
            #print(list(prev_obj_list[i].values())[0])
            matrix[i][j] = get_distance(list(prev_obj_list[i].values())[0], cur_obj_list[j])

    arg_min = np.argmin(matrix, axis = 1)

    cur_used = np.zeros(len(cur_obj_list))

    for idx, prev_obj in enumerate(prev_obj_list):
        prev_id, prev_bb = list(prev_obj.items())[0]
        if matrix[idx][arg_min[idx]] < threshold:
            new_obj = {prev_id:cur_obj_list[arg_min[idx]]}
            cur_used[arg_min[idx]] = 1
        else:
            new_obj = {prev_id:prev_bb}
        object_list.append(new_obj)
    # print("prev : ", prev_obj_list)
    # print("cur : ", cur_obj_list)
    # print("arg_min : ", arg_min)
    # print("cur_used : ", cur_used)
    for i in range(len(cur_used)):
        if cur_used[i] == 0 and max_num < 6:
            cur_used[i] = 1
            cur_id = max_num
            new_obj = {cur_id:cur_obj_list[i]}
            object_list.append(new_obj)
            max_num += 1

    if len(object_list) > 3:
        object_list = object_list[:3]
    return max_num, object_list

def object_tracking(max_num, prev_obj_list, cur_obj_list):
    matrix = np.zeros((len(prev_obj_list), len(cur_obj_list)))
    for i in range(len(prev_obj_list)):
        for j in range(len(cur_obj_list)):
            #print(list(prev_obj_list[i].values())[0])
            matrix[i][j] = get_distance(list(prev_obj_list[i].values())[0], cur_obj_list[j])

    arg_min = np.argmin(matrix, axis = 1)

    if len(prev_obj_list) == len(cur_obj_list):
        object_list = []
        cur_visited = np.zeros((len(cur_obj_list)))
        #print("arg_min : ", arg_min)
        for idx, obj in enumerate(prev_obj_list):
            #new_obj = obj.copy()
            new_obj = {}
            id, _ = list(obj.items())[0]
            # print("id : ", id)
            if cur_visited[arg_min[idx]] == 0:
                cur_visited[arg_min[idx]] = 1
                new_obj[id] = cur_obj_list[arg_min[idx]]
            else:
                min = 100000000
                ii = -1
                for jw in range(len(cur_visited)):    
                    if cur_visited[jw] == 0 and min > matrix[idx][jw]:
                        ii = jw
                        min = matrix[idx][jw]
                cur_visited[ii] = 1
                #print("ii : ", ii)
                new_obj[id] = cur_obj_list[ii]

            object_list.append(new_obj)
        
    # case 2 : prev obj number > cur obj number
    elif len(prev_obj_list) > len(cur_obj_list):
        
        object_list = []
        id = -1 * np.ones(len(cur_obj_list))
        value = -1 * np.ones(len(cur_obj_list))

        for idx, min_idx in enumerate(arg_min):
            if id[min_idx] == -1 and value[min_idx] == -1:
                # cur_obj = prev_obj_list[idx].copy() # dict = {id, [xywh]}
                cur_id = list(prev_obj_list[idx].keys())[0] # id
                bb = cur_obj_list[min_idx] # [x, y, w, h]
                
                # cur_obj['cur_box'] = bb
                cur_obj = {}
                cur_obj[cur_id] = bb
                id[min_idx] = cur_id
                value[min_idx] = matrix[idx][min_idx]

                object_list.append(cur_obj)
                #print("init : ", object_list)
            else:
                # add another object with larger id number 
                if matrix[idx][min_idx] < value[min_idx]:
                    origin_id = id[min_idx]
                    for obj in object_list:
                        #if obj['id'] == origin_id:
                        if list(obj.keys())[0] == origin_id:
                            object_list.remove(obj)
                    #print("after remove : ", object_list)

                    #new_obj = prev_obj_list[idx].copy()
                    new_obj = {}
                    new_id = list(prev_obj_list[idx].keys())[0]
                    #new_obj['cur_box'] = cur_obj_list[min_idx]
                    new_obj[new_id] = cur_obj_list[min_idx]

                    object_list.append(new_obj)

                    id[min_idx] = int(new_id)
                    value[min_idx] = matrix[idx][min_idx]

    # case 3 : prev obj number < cur obj number
    else: # len(prev_obj_list) < len(cur_obj_list)
        object_list = []
        cur_visited = np.zeros((len(cur_obj_list)))

        # origin 
        for idx, obj in enumerate(prev_obj_list):
            #new_obj = obj.copy()
            new_obj = {}
            new_id = list(obj.keys())[0]
            new_obj[new_id] = cur_obj_list[arg_min[idx]]
            object_list.append(new_obj)
            
            cur_visited[arg_min[idx]] = 1

        # add additional object here
        for idx in range(len(cur_visited)):
            if cur_visited[idx] == 0:
                added_obj = {max_num:cur_obj_list[idx]}
                #added_obj = {'id' : max_num, 'cur_box' : cur_obj_list[idx]}
                max_num += 1
                object_list.append(added_obj)

    return max_num, object_list

def inference_(frame, output_dict, net):
    
    inf = {}

    for i in range(frame - net.args.input, frame):
            for obj in output_dict[str(i)]:
                id = list(obj.keys())[0]
                xywh = list(obj.values())[0]

                if id in inf.keys():
                    inf[id] = np.vstack((inf[id], np.array(xywh)))
                else:
                    inf[id] = np.array(xywh)

    # exception num of value is less than 16
    for id, pos in inf.items():
        if pos.shape[0] != 16:
            #print("*"*50)
            last = pos.shape[0] # 10
            for i in range(last, net.args.input):
                pos = np.vstack((pos, pos[-1]))
            inf[id] = pos

    #print("inf : ", inf)
    ret = {}
    for id, pos in inf.items():
        
        spd = pos[1:] = pos[:-1]
        pos_input = torch.tensor(pos).unsqueeze(0)
        pos_input = pos_input.type(torch.float32)
        pos_input = pos_input.to(device='cuda')

        spd_input = torch.tensor(spd).unsqueeze(0)
        spd_input = spd_input.type(torch.float32)
        spd_input = spd_input.to(device='cuda')

        with torch.no_grad():
            #print("inference")
            #_, _, intentions = net(speed=spd_input, pos=pos_input, average=True)
            _, _, intentions = net(speed=spd_input, pos=pos_input)
            #print("person intentions : ", intentions)
            intentions = intentions.detach().cpu().numpy()
        threshold = 0.52
        
        if intentions[0][1] > threshold:
            #print("up")
            intention = 1
        else:
            #print("down")
            intention = 0
        #print(id, ": ", intention)
        ret[id] = intention
    return ret



def detect_object_number(mask):
    img = mask.copy()
    img[img > 190] = 255
    img[img[:,:,0]<100] = 0
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((2,2),np.uint8), iterations =2)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((2,2),np.uint8), iterations =2)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((2,2),np.uint8), iterations =2)

    thresh = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    contours,hierarchy = cv2.findContours(thresh, 1, 2)
    num = 0
    for item in range(len(contours)):
        cnt = contours[item]
        if len(cnt)>5:
            num += 1
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
    return num

def detect_color(img):
    # import pdb; pdb.set_trace()
    print(img.shape)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = outline2black(img)
    color_level = {}
    color_cnt = {}
    for color in ['white','red', 'green']:
        if color == 'white':
            #https://stackoverflow.com/questions/22588146/tracking-white-color-using-python-opencv       
            sensitivity = 255
            lower = np.array([0,5,50])
            upper = np.array([179,50,255])
        elif color == 'red':
            #lower = np.array([0, 70, 50])
            lower = np.array([0,40,90])
            #upper = np.array([179, 255, 255])
            upper = np.array([255,255,255])
        elif color == 'green':
            lower = np.array([(54, 43, 95)])
            upper = np.array([164, 225,225])
            
        masking = cv2.inRange(hsv, lower, upper)
        detect = cv2.bitwise_and(img,img, mask=masking)
        cnt = 0
        if color == 'white':
            detect2 = np.zeros_like(detect[:,:,0])
            ret1, image_process = cv2.threshold(detect,100,255,cv2.THRESH_BINARY)
            for ch in range(3):
                detect2[image_process[:,:,ch]==255] = 255
            detect2[(image_process[:,:,0]-image_process[:,:,1]) > 20] = 0
            detect2[(image_process[:,:,0]-image_process[:,:,2]) > 20] = 0
            detect2[(image_process[:,:,1]-image_process[:,:,2]) > 20] = 0
            detect = detect2
        elif color == 'red':
            #cnt_img = detect.copy()
            #  cnt = detect_object_number(detect)
            detect[detect > 190] = 255
            detect[:,:,1:3] = 0
            detect = cv2.morphologyEx(detect, cv2.MORPH_OPEN, np.ones((2,2),np.uint8), iterations =1)
        elif color == 'green':
            detect[:,:,0] = 0
            detect[:,:,2] = 0
            detect[detect > 50] = 255
            detect = cv2.morphologyEx(detect, cv2.MORPH_OPEN, np.ones((2,2),np.uint8), iterations =1)
        level = detect.mean()
        color_level.update({color:level})
        color_cnt.update({color:cnt})
    # if color_level['white'] < color_level['red']:
    #     try:
    #         cnt = detect_object_number(detect)
    #     except:
    #         cnt = 0
    #     color_cnt.update({'red':cnt})

    return color_level, color_cnt

def outline2black(img):
    test = img.copy()
    # new_image = img.copy()
    gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]
    w, h = thresh.shape
    mask1 = np.ones_like(thresh)
    mask2 = np.ones_like(thresh)
    
    mask1[0:w//15,:] = 0
    mask1[:,0:h//15] = 0
    mask1[w-w//15:,:] = 0
    mask1[:,h-h//15:] = 0
    
    
    max_w = w//15
    max_h = h//15
    for h_idx in range(h-1):            
        for w_idx in range(w-1):
            if thresh[w_idx,h_idx] == 0 and thresh[w_idx+1,h_idx] == 0:
                mask1[w_idx,h_idx] =0
            else:
                break
            if w_idx > max_w:
                break
            
        for inverse_w_idx in range(w-1,0,-1):
            if thresh[inverse_w_idx,h_idx] ==0 and thresh[inverse_w_idx-1,h_idx] == 0:
                mask1[inverse_w_idx,h_idx] = 0
            else:
                break
            if w-inverse_w_idx-1 > max_w:
                break
            
    for w_idx in range(w-1):
        for h_idx in range(h-1):
            if thresh[w_idx,h_idx] == 0 and thresh[w_idx,h_idx+1] == 0:
                    mask1[w_idx,h_idx] =0
            else:
                break
            if h_idx > max_h:
                break
            
        for inverse_h_idx in range(h-1,0,-1):
            if thresh[w_idx,inverse_h_idx] ==0 and thresh[w_idx,inverse_h_idx-1] == 0:
                mask1[w_idx,inverse_h_idx] = 0
            else:
                break    
            if h-inverse_h_idx-1 > max_h:
                break
    
    
    img = cv2.bitwise_and(img, img, mask=mask1)
    for w_idx in range(w):
        for h_idx in range(h-1):
            if thresh[w_idx,h_idx] == 0 or thresh[w_idx,h_idx+1] == 1 :
                break
            else:
                mask2[w_idx,h_idx] = 0
        for inverse in range(h-1,0,-1):
            if thresh[w_idx,inverse] == 0  or thresh[w_idx,inverse-1] == 1:
                break
            else:
                mask2[w_idx,inverse] = 0
            
    for h_idx in range(h):
        for w_idx in range(w-1):
            if thresh[w_idx,h_idx] == 0 or thresh[w_idx+1,h_idx] == 1 :
                break
            else:
                mask2[w_idx,h_idx] = 0
        for inverse in range(w-1,0,-1):
            if thresh[inverse,h_idx] == 0 or thresh[inverse-1, h_idx]== 1:
                break
            else:
                mask2[inverse,h_idx] = 0
    
    img_out = cv2.bitwise_and(img, img, mask=mask2)
    return img_out
