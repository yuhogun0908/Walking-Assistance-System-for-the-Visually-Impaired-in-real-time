import yaml
import argparse
import os
from utils_general import *
from detection import *
import network

class args():
    def __init__(self):
        #self.jaad_dataset = '/data/smailait-data/JAAD/processed_annotations' #folder containing parsed jaad annotations (used when first time loading data)
        self.jaad_dataset = '../JAAD/processed_annotations'
        
        self.dtype        = 'test'
        self.from_file    = False #read dataset from csv file or reprocess data
        self.save         = True
        
        #self.file         = '/data/smailait-data/jaad_test_16_16.csv'
        self.file         = './jaad_test_16_16.csv'
        
        #self.save_path    = '/data/smailait-data/jaad_test_16_16.csv'
        self.save_path    = './jaad_test_16_16.csv'
        
        #self.model_path    = '/data/smailait-data/models/multitask_pv_lstm_trained.pkl'
        self.model_path    = './models/multitask_pv_lstm_trained_1000.pkl'
        
        self.loader_workers = 10
        self.loader_shuffle = True
        self.pin_memory     = False
        self.image_resize   = [240, 426]
        self.device         = 'cuda'
        
        #self.batch_size     = 100
        self.batch_size     = 1
        
        self.n_epochs       = 100
        self.hidden_size    = 512
        self.hardtanh_limit = 100
        self.input  = 16
        self.output = 16
        self.stride = 8
        self.skip   = 1
        self.task   = 'bounding_box-intention'
        self.use_scenes = False       
        self.lr = 0.00001

argv = args()

def run(args):

    if args.video2frames:
        videoRootdir = 'videos'
        resize = args.resize
        size = tuple((args.size, args.size))
        print('Start Video to Images')
        video2frames(videoRootdir,args.videoName,resize,size)
        print('Finish')

    # object detection
    

    net = network.PV_LSTM(argv).to(argv.device)
    net.eval()

    object_detect(args, net)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', type=int, default=1, help='Whether use GPU')
    parser.add_argument('--videoName', type=str, default='Final1.mp4')
    parser.add_argument('--resize', type=bool, default=False)
    parser.add_argument('--size', type=int, default=1280)
    parser.add_argument('--video2frames', type=bool, default=True)
    
    #object detection
    parser.add_argument('--device', type = str, default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--weights', type=str, default='yolormodel/yolor_p6.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--names', type=str, default='yolormodel/data/coco.names', help='*.cfg path')
    parser.add_argument('--cfg', type=str, default='yolormodel/yolor_p6.cfg', help='*.cfg path')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    # parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')


    args= parser.parse_args()

    run(args)
