from django.shortcuts import render
from django.http import StreamingHttpResponse

import yolov5,torch
from yolov5.utils.general import (check_img_size, non_max_suppression, scale_coords, 
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import cv2
from PIL import Image as im
# Create your views here.
def index(request):
    return render(request,'index.html')
print(torch.cuda.is_available())

#load model
model = yolov5.load('best-fix.pt')

#deklarasi var
count_Mobil = 0
data_Mobil = []
count_Motor = 0
data_Motor = []
count_Bus = 0
data_Bus = []
count_Truk = 0
data_Truk = []
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
device = select_device('') # 0 for gpu, '' for cpu
# initialize deepsort
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")

deepsort = DeepSort('osnet_x0_25',
                    device,
                    max_dist=cfg.DEEPSORT.MAX_DIST,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    )
# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names

def stream():
    cap = cv2.VideoCapture(0)
    model.conf = 0.5
    model.iou = 0.7
    model.classes = [0,1,2,3]
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to capture image")
            break

        results = model(frame, augment=True)
        # proccess
        annotator = Annotator(frame, line_width=2, pil=not ascii) 
        
        det = results.pred[0]
        if det is not None and len(det):   
            xywhs = xyxy2xywh(det[:, 0:4])
            confs = det[:, 4]
            clss = det[:, 5]
            outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), frame)
            if len(outputs) > 0:
                for j, (output, conf) in enumerate(zip(outputs, confs)):

                    bboxes = output[0:4]
                    id = output[4]
                    cls = output[5]
                    #count
                    count_obj(bboxes,w,h,id,cls)
                    c = int(cls)  # integer class
                    label = f'{id} {names[c]} {conf:.2f}'
                    annotator.box_label(bboxes, label, color=colors(c, True))

        else:
            deepsort.increment_ages()
        
        im0 = annotator.result()
        w, h = im0.shape[1],im0.shape[0]
        if check_imshow:
                global count_Mobil
                color=(0,255,0) #warna hijau
                color_text=(0,0,0) #warna hitam
                #garis horizontal
                start_point = (0, h-150) 
                end_point = (w, h-150)
                #garis vertikal
                #start_point = (w-350, 0) 
                #end_point = (w-350, h)
                cv2.line(im0, start_point, end_point, color, thickness=2) 
                thickness = 1
                org1 = (50, 30)
                org2 = (50, 50)                
                org3 = (450, 30)
                org4 = (450, 50)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.5
                cv2.putText(im0, "Jumlah Motor = " + str(count_Motor), org1, font, 
                fontScale, color_text, thickness, cv2.LINE_AA)
                cv2.putText(im0, "Jumlah Mobil = " + str(count_Mobil), org2, font, 
                fontScale, color_text, thickness, cv2.LINE_AA)
                cv2.putText(im0, "Jumlah Bus = " + str(count_Bus), org3, font, 
                fontScale, color_text, thickness, cv2.LINE_AA)
                cv2.putText(im0, "Jumlah Truk = " + str(count_Truk), org4, font, 
                fontScale, color_text, thickness, cv2.LINE_AA)   
        image_bytes = cv2.imencode('.jpg', im0)[1].tobytes() #mengubah frame gambar ke byte supaya dapat dikirimkan ke browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n')  

def video_feed(request):
    return StreamingHttpResponse(stream(), content_type='multipart/x-mixed-replace; boundary=frame')

def count_obj(box,w,h,id,cls):
    global count_Mobil,data_Mobil,count_Motor,data_Motor,count_Bus,data_Bus,count_Truk,data_Truk
    center_coordinates = (int(box[0]+(box[2]-box[0])/2) , int(box[1]+(box[3]-box[1])/2))
    if cls == 0: #motor
        if int(box[1]+(box[3]-box[1])/2) > (h-150): #ini untuk horizontal
        #if int(box[0]+(box[2]-box[0])/2) < (w-350): # vertikal
            if id not in data_Motor:
                count_Motor += 1
                data_Motor.append(id)
    if cls == 1: #mobil
        if int(box[1]+(box[3]-box[1])/2) > (h-150): #ini untuk horizontal
        #if int(box[0]+(box[2]-box[0])/2) < (w-350): #vertikal
            if id not in data_Mobil:
                count_Mobil += 1
                data_Mobil.append(id)
    if cls == 2: #bus
        if int(box[1]+(box[3]-box[1])/2) > (h-150): #ini untuk horizontal
        # if int(box[0]+(box[2]-box[0])/2) < (w-350): #vertikal
            if id not in data_Bus:
                count_Bus += 1
                data_Bus.append(id)
    if cls == 3: #truk
        if int(box[1]+(box[3]-box[1])/2) > (h-150): #ini untuk horizontal
        #if int(box[0]+(box[2]-box[0])/2) < (w-350): # vertikal
            if id not in data_Truk:
                count_Truk += 1
                data_Truk.append(id)    