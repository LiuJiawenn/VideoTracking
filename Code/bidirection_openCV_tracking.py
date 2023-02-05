from Code.xml_analyse import XMLAnalyse
from tracking import Tracking
import os
import cv2 as cv


##############################################################################################################
# I use this script to realize bidirection openCV tracking
# step 1: use function reverse_a_video in reverse.py to creat a reversed video
# step 2: use function reverse_annotation_files in reverse.py to prepare annotation files for the reversed video
# step 3: settle down paramaters below and run this script

# which openCV algo you want to use
algo = 'csrt'
# path for original video
video_path1 = '../video data/s8_r5_a16.avi'
# path for reversed video
video_path2 = '../video data/reverse_s8_r5_a16.avi'
# annotation files for original video
f_path1 = '../label manual/interval fixed'
# annotation files for reversed video
f_path2 = '../label manual/reverse'
# where you want to save the tracking result
s_path = '../result/reverse'
################################################################################################################

def showBndbox(frame, bndbox, color):
    cv.line(frame, (int(bndbox[0]), int(bndbox[2])), (int(bndbox[0]), int(bndbox[3])), color, 2)  # left
    cv.line(frame, (int(bndbox[1]), int(bndbox[2])), (int(bndbox[1]), int(bndbox[3])), color, 2)  # right
    cv.line(frame, (int(bndbox[0]), int(bndbox[2])), (int(bndbox[1]), int(bndbox[2])), color, 2)  # top
    cv.line(frame, (int(bndbox[0]), int(bndbox[3])), (int(bndbox[1]), int(bndbox[3])), color, 2)  # bottom
    cv.imshow('frame', frame)
    cv.waitKey(1)


for interval in [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100, 200, 1000]:
    folder_path1 = f_path1 + '/interval ' + str(interval)
    folder_path2 = f_path2 + '/interval ' + str(interval)
    save_path = s_path + '/' + algo + '/interval ' + str(interval)
    if not os.path.exists(save_path):  # create directory
        os.makedirs(save_path)
    bndbox_list = os.listdir(folder_path1)
    bndbox_list.sort(key=lambda x: int(x[0:-4]))

    boxes = []
    box1 = Tracking.openCV_tracking(video_path=video_path1, folder_path=folder_path1, save_flag=False, algo_flag=algo)
    box2 = Tracking.openCV_tracking(video_path=video_path2, folder_path=folder_path2, save_flag=False, algo_flag=algo)
    box2.reverse()

    for i in range(len(box1)):
        if (i == 0) | (i == (len(box1)-1)) | ((i+1) % interval == 0):
            boxes.append(box1[i])
            continue
        center1 = [(box1[i][0] + box1[i][1]) / 2, (box1[i][2] + box1[i][3]) / 2]
        center2 = [(box2[i][0] + box2[i][1]) / 2, (box2[i][2] + box2[i][3]) / 2]
        size1 = [(box1[i][1] - box1[i][0]) / 2, (box1[i][3] - box1[i][2]) / 2]
        size2 = [(box2[i][1] - box2[i][0]) / 2, (box2[i][3] - box2[i][2]) / 2]
        rate = ((i+1) % interval)/interval
        size = [(1-rate)*size1[0]+rate*size2[0], (1-rate)*size1[1]+rate*size2[1]]
        center = [(1-rate)*center1[0]+rate*center2[0], (1-rate)*center1[1]+rate*center2[1]]
        box = [0, 0, 0, 0]
        box[0] = int(center[0]-size[0])
        box[1] = int(center[0]+size[0])
        box[2] = int(center[1]-size[1])
        box[3] = int(center[1]+size[1])
        boxes.append(box.copy())

    for i in range(len(boxes)):
        xmladdress = save_path + "/" + str(i + 1) + ".xml"
        XMLAnalyse.writeXML(xmladdress, boxes[i])

    cap = cv.VideoCapture(video_path1)
    for i in range(len(boxes)):
        ret, frame = cap.read()
        if ret is False:
            exit()
        showBndbox(frame, boxes[i], [212, 237, 244])
    cap.release()
