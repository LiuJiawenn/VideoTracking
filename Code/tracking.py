import numpy as np
import cv2 as cv
import os
from xml_analyse import XMLAnalyse


class Tracking:
    """ This class include 10 tracking algorithme, they are:
    ['linear', 'opticalFlow', 'bidirection', 'csrt', 'medianflow', 'mosse', 'mil', 'boosting', 'kcf', 'tld']

    You can use callalgobyname to call an algorithme by the string above, all call its own function.
    All function have the same parameters:
    video_path: the video data path ,
    folder_path: directory for pre annotation files,
    color: color for bounding box in RGB,
    show_flag: if you want to see the tracking result on video, make it 1.Or you can only get a list of bndbox
    save_flag: if you want save tracking result in XML files,make it 1
    save_path: where you want to save tracking result"""

    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.02))

    feature_params = dict(maxCorners=1000000, qualityLevel=0.01, minDistance=1, blockSize=2)

    @staticmethod
    def callalgobyname(save_path, folder_path, video_path='../video data/s8_r5_a16.avi', algo='linear', color=[212, 237, 244], show_flag=1,
                       save_flag=1):
        if algo == 'linear':
            Tracking.linear_tracking(video_path=video_path, folder_path=folder_path, color=color,
                                     show_flag=show_flag, save_flag=save_flag, save_path=save_path)
        elif algo == 'opticalFlow':
            Tracking.optical_flow_tracking(video_path=video_path, folder_path=folder_path, color=color,
                                           show_flag=show_flag, save_flag=save_flag, save_path=save_path)
        elif algo == 'bidirection':
            Tracking.bidirection_optical_flow_tracking(video_path=video_path, folder_path=folder_path, color=color,
                                                       show_flag=show_flag, save_flag=save_flag, save_path=save_path)
        elif algo in ['csrt', 'medianflow', 'mosse', 'mil', 'boosting', 'kcf', 'tld']:
            Tracking.openCV_tracking(algo_flag=algo, video_path=video_path, folder_path=folder_path, color=color,
                                     show_flag=show_flag, save_flag=save_flag, save_path=save_path)
        else:
            print("No this algo, please check your spelling")

    @staticmethod
    def linear_tracking(video_path='../video data/s8_r5_a16.avi',
                        folder_path='../label manual/interval fixed/interval 20',
                        color=[212, 237, 244],
                        show_flag=1,
                        save_flag=1,
                        save_path='../result/interval fixed/linear/interval 20'):
        # Read pre-labeling
        bndbox_list = os.listdir(folder_path)  # Find all the video annotation files under the folder
        bndbox_list.sort(key=lambda x: int(x[0:-4]))  # Sort by numerical order
        boxes = []

        for i in range(len(bndbox_list) - 1):
            bndbox1 = XMLAnalyse.analyseXML(folder_path + '/' + bndbox_list[i])
            bndbox2 = XMLAnalyse.analyseXML(folder_path + '/' + bndbox_list[i + 1])
            frameinterval = int(bndbox_list[i + 1][0:-4]) - int(bndbox_list[i][0:-4])
            changes = [(bndbox2[k] - bndbox1[k]) / frameinterval for k in range(len(bndbox1))]

            # read one frame correspond to the bndbox1
            bndbox = bndbox1.copy()
            boxes.append(bndbox.copy())

            # arrange the bndbox between bndbox1 and bndbox2
            for j in range(frameinterval - 1):
                bndbox = [bndbox[k] + changes[k] for k in range(len(bndbox))]
                boxes.append(bndbox.copy())

        # the last frame
        bndbox = XMLAnalyse.analyseXML(folder_path + '/' + bndbox_list[-1])
        boxes.append(bndbox.copy())

        if show_flag:
            cap = cv.VideoCapture(video_path)
            for i in range(len(boxes)):
                ret, frame = cap.read()
                if ret is False:
                    exit()
                Tracking.__showBndbox(frame, boxes[i], color)
            cap.release()

        if save_flag:
            for i in range(len(boxes)):
                xmladdress = save_path + "/" + str(i + 1) + ".xml"
                XMLAnalyse.writeXML(xmladdress, boxes[i])

        cv.destroyAllWindows()
        return boxes

    @staticmethod
    def optical_flow_tracking(video_path='../video data/s8_r5_a16.avi',
                              folder_path='../label manual/interval fixed/interval 20',
                              color=[212, 237, 244],
                              show_flag=1,
                              save_flag=1,
                              save_path='../result/interval fixed/opticalFlow/interval 20'):

        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print("Could not open video")
            exit()

        # Find all the video annotation files under the folder and Sort by numerical order
        bndbox_list = os.listdir(folder_path)
        bndbox_list.sort(key=lambda x: int(x[0:-4]))

        # make box size change between two pre-label
        sizelist = Tracking.__cal_box_size_list(folder_path, bndbox_list)
        # inport linear tracking result as default
        linear_boxes = Tracking.linear_tracking(folder_path=folder_path, show_flag=0, save_flag=0)

        bndbox_index = 0
        frame_index = 0
        boxes = []

        # read the first frame and transform to gray image
        ret, old_frame = cap.read()
        frame_index = frame_index + 1
        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

        # the first bndbox
        bndbox = XMLAnalyse.analyseXML(folder_path + '/' + bndbox_list[bndbox_index])
        bndbox_index = bndbox_index + 1
        boxes.append(bndbox.copy())

        # initial the key points inside the first bndbox
        p0 = Tracking.__settle_feature_points(bndbox, old_frame)

        # optical flow
        while True:
            ret, frame = cap.read()
            if ret is False:
                break
            frame_index = frame_index + 1

            # when it comes to a pre-labelled frame
            if frame_index == int(bndbox_list[bndbox_index][0:-4]):
                mask = np.zeros_like(old_frame)
                bndbox = XMLAnalyse.analyseXML(folder_path + '/' + bndbox_list[bndbox_index])
                boxes.append(bndbox.copy())
                p1 = Tracking.__settle_feature_points(bndbox, frame)
                p0 = p1.copy()
                if bndbox_index < len(bndbox_list) - 1:
                    bndbox_index = bndbox_index + 1
                continue

            # calcul potical flow
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            if len(p0) > 9:
                p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **Tracking.lk_params)
            else:
                # Abandon optical flow and start linear tracking when an unsuitable frame is encountered
                bndbox = linear_boxes[frame_index - 1]
                p1 = Tracking.__settle_feature_points(bndbox, frame)
                p0 = p1.copy()
                boxes.append(bndbox.copy())
                continue

            # Selection based on state.
            # st==1 A state of 1 means that the feature point of the previous frame was found in the second frame
            good_new = p1[st == 1]

            # calcul key points center
            center_X_good_new = np.mean(good_new[:, 0])
            center_Y_good_new = np.mean(good_new[:, 1])

            if frame_index <= len(sizelist):
                xchange = int(sizelist[frame_index - 1][0] / 2)
                ychange = int(sizelist[frame_index - 1][1] / 2)
                bndbox[0] = int(center_X_good_new - xchange)
                bndbox[1] = int(center_X_good_new + xchange)
                bndbox[2] = int(center_Y_good_new - ychange + 5)
                bndbox[3] = int(center_Y_good_new + ychange + 5)

            else:
                xchange = center_X_good_new - (bndbox[1] + bndbox[0]) / 2
                ychange = center_Y_good_new - (bndbox[3] + bndbox[2]) / 2
                bndbox[0] = int(bndbox[0] + xchange)
                bndbox[1] = int(bndbox[1] + xchange)
                bndbox[2] = int(bndbox[2] + ychange + 5)
                bndbox[3] = int(bndbox[3] + ychange + 5)

            boxes.append(bndbox.copy())

            xmin = bndbox[0]
            xmax = bndbox[1]
            ymin = bndbox[2]
            ymax = bndbox[3]

            points = []
            for point in good_new:
                if (point[0] >= xmin) & (point[0] <= xmax) & (point[1] >= ymin) & (point[1] <= ymax):
                    points.append(point)
            p0 = np.array(points).reshape(-1, 1, 2)

            old_gray = frame_gray.copy()
            if len(p0) < 10:
                new_point = Tracking.__settle_feature_points(bndbox, frame)
                p0 = np.append(p0, new_point, axis=0)

        cap.release()
        if show_flag:
            cap = cv.VideoCapture(video_path)
            for i in range(len(boxes)):
                ret, frame = cap.read()
                if ret is False:
                    break
                Tracking.__showBndbox(frame, boxes[i], color)
            cap.release()
            cv.destroyAllWindows()

        if save_flag:
            for i in range(len(boxes)):
                xmladdress = save_path + "/" + str(i + 1) + ".xml"
                XMLAnalyse.writeXML(xmladdress, boxes[i])
        return boxes

    @staticmethod
    def bidirection_optical_flow_tracking(video_path='../video data/s8_r5_a16.avi',
                                          folder_path='../label manual/interval fixed/interval 20',
                                          color=[212, 237, 244],
                                          show_flag=1,
                                          save_flag=1,
                                          save_path='../result/interval fixed/bidirection/interval 20'):
        # read video
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print("Could not open video")
            exit()

        # Find all the video annotation files under the folder and sort by numerical order
        bndbox_list = os.listdir(folder_path)
        bndbox_list.sort(key=lambda x: int(x[0:-4]))

        sizelist = Tracking.__cal_box_size_list(folder_path, bndbox_list)
        linear_boxes = Tracking.linear_tracking(folder_path=folder_path, show_flag=0, save_flag=0)
        bndbox_index = 0
        frame_index = 0
        boxes = []

        # read the first frame
        ret, old_frame = cap.read()
        frame_index = frame_index + 1
        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
        bndbox = XMLAnalyse.analyseXML(folder_path + '/' + bndbox_list[bndbox_index])
        bndbox_index = bndbox_index + 1
        boxes.append(bndbox.copy())

        # initial the key points inside the first bndbox
        p0 = Tracking.__settle_feature_points(bndbox, old_frame)

        # optical flow
        while True:
            ret, frame = cap.read()
            if ret is False:
                break
            frame_index = frame_index + 1

            # when it comes to a pre-labeling frame
            if frame_index == int(bndbox_list[bndbox_index][0:-4]):
                mask = np.zeros_like(old_frame)
                bndbox = XMLAnalyse.analyseXML(folder_path + '/' + bndbox_list[bndbox_index])
                boxes.append(bndbox.copy())
                p1 = Tracking.__settle_feature_points(bndbox, frame)
                p0 = p1.copy()

                if bndbox_index < len(bndbox_list) - 1:
                    bndbox_index = bndbox_index + 1
                continue

            # calcul potical flow
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            if len(p0) < 12:
                bndbox = linear_boxes[frame_index - 1]
                p1 = Tracking.__settle_feature_points(bndbox, frame)
                p0 = p1.copy()
                boxes.append(bndbox.copy())
                continue
            # # The keypoint of the previous frame and the image of the current frame are used as input to get the
            # position of the keypoint in the current frame
            p1, st1, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None,
                                                   **Tracking.lk_params)
            # Key points and images of the current frame and the previous frame as input
            p0r, st2, err = cv.calcOpticalFlowPyrLK(frame_gray, old_gray, p1, None, **Tracking.lk_params)
            # compare the position between retraced key points and true key points in current image
            d = abs(p0 - p0r).reshape(-1, 2).max(-1).reshape(-1, 1)
            # Determine whether the value in d is less than 1, a trace greater than 1 is considered a wrong trace point
            # good.reshape(-1, 1)
            good_new = p1[d < 1]
            st1 = st1[d < 1]
            # st==1 A state of 1 means that the feature point of the previous frame was found in the second frame
            good_new = good_new[st1 == 1]

            if len(good_new) == 0:
                good_new = Tracking.__settle_feature_points(bndbox, frame)
                good_new = good_new.reshape(-1, 2)

            center_X_good_new = np.mean(good_new[:, 0])
            center_Y_good_new = np.mean(good_new[:, 1])

            if frame_index <= len(sizelist):
                xchange = int(sizelist[frame_index - 1][0] / 2)
                ychange = int(sizelist[frame_index - 1][1] / 2)
                bndbox[0] = int(center_X_good_new - xchange)
                bndbox[1] = int(center_X_good_new + xchange)
                bndbox[2] = int(center_Y_good_new - ychange + 5)
                bndbox[3] = int(center_Y_good_new + ychange + 5)

            else:
                xchange = center_X_good_new - (bndbox[1] + bndbox[0]) / 2
                ychange = center_Y_good_new - (bndbox[3] + bndbox[2]) / 2
                bndbox[0] = int(bndbox[0] + xchange)
                bndbox[1] = int(bndbox[1] + xchange)
                bndbox[2] = int(bndbox[2] + ychange + 5)
                bndbox[3] = int(bndbox[3] + ychange + 5)

            boxes.append(bndbox.copy())

            xmin = bndbox[0]
            xmax = bndbox[1]
            ymin = bndbox[2]
            ymax = bndbox[3]

            points = []
            for point in good_new:
                if (point[0] >= xmin) & (point[0] <= xmax) & (point[1] >= ymin) & (point[1] <= ymax):
                    points.append(point)
            p0 = np.array(points).reshape(-1, 1, 2)

            old_gray = frame_gray.copy()
            if len(p0) < 10:
                new_point = Tracking.__settle_feature_points(bndbox, frame)
                p0 = np.append(p0, new_point, axis=0)

        cap.release()
        if show_flag:
            cap = cv.VideoCapture(video_path)
            for i in range(len(boxes)):
                ret, frame = cap.read()
                if ret is False:
                    break
                Tracking.__showBndbox(frame, boxes[i], color)
            cap.release()
            cv.destroyAllWindows()

        if save_flag:
            for i in range(len(boxes)):
                xmladdress = save_path + "/" + str(i + 1) + ".xml"
                XMLAnalyse.writeXML(xmladdress, boxes[i])

        return boxes

    @staticmethod
    def openCV_tracking(video_path='../video data/s8_r5_a16.avi',
                        folder_path='../label manual/interval fixed/interval 20',
                        color=[212, 237, 244],
                        algo_flag='csrt',
                        show_flag=1,
                        save_flag=1,
                        save_path='../result/interval fixed/csrt/interval 20'):

        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print("Could not open video")
            exit()

        OPENCV_OBJECT_TRACKERS = {
            "csrt": cv.TrackerCSRT_create,
            "kcf": cv.TrackerKCF_create,
            "boosting": cv.legacy.TrackerBoosting_create,
            "mil": cv.TrackerMIL_create,
            "tld": cv.legacy.TrackerTLD_create,
            "medianflow": cv.legacy.TrackerMedianFlow_create,
            "mosse": cv.legacy.TrackerMOSSE_create
        }

        bndbox_list = os.listdir(folder_path)
        bndbox_list.sort(key=lambda x: int(x[0:-4]))
        linear_boxes = Tracking.linear_tracking(folder_path=folder_path, show_flag=0, save_flag=0)
        bndbox_index = 0
        frame_index = 0
        boxes = []

        # Initialize the position of the bndbox
        bndbox = XMLAnalyse.analyseXML(folder_path + '/' + bndbox_list[0])
        bndbox_index = bndbox_index + 1
        boxes.append(bndbox.copy())
        tracking_box = [bndbox[0], bndbox[2], bndbox[1] - bndbox[0], bndbox[3] - bndbox[2]]
        tracker = OPENCV_OBJECT_TRACKERS[algo_flag]()

        # Read first frame.
        ok, frame = cap.read()
        frame_index = frame_index + 1

        # Initialize tracker with first frame and bounding box
        tracker.init(frame, tracking_box)

        while True:
            # Read a new frame
            ok, frame = cap.read()
            if not ok:
                break
            frame_index = frame_index + 1
            # when it comes to a pre-labeling frame
            if frame_index == int(bndbox_list[bndbox_index][0:-4]):
                bndbox = XMLAnalyse.analyseXML(folder_path + '/' + bndbox_list[bndbox_index])
                tracking_box = [bndbox[0], bndbox[2], bndbox[1] - bndbox[0], bndbox[3] - bndbox[2]]
                boxes.append(bndbox.copy())
                tracker = OPENCV_OBJECT_TRACKERS[algo_flag]()
                tracker.init(frame, tracking_box)
                if bndbox_index < len(bndbox_list) - 1:
                    bndbox_index = bndbox_index + 1
                continue

            # Update tracker
            ok, bndbox1 = tracker.update(frame)
            if ok:
                tracking_box = [int(bndbox1[0]), int(bndbox1[1]), int(bndbox1[2]), int(bndbox1[3])]
                bndbox = [tracking_box[0], tracking_box[0] + tracking_box[2], tracking_box[1],
                          tracking_box[1] + tracking_box[3]]
            else:
                # Abandon optical flow and start linear tracking when an unsuitable frame is encountered
                bndbox = linear_boxes[frame_index - 1]
                tracking_box = [bndbox[0], bndbox[2], bndbox[1] - bndbox[0], bndbox[3] - bndbox[2]]

            # Tracking success, Draw bounding box
            boxes.append(bndbox.copy())

        cap.release()
        if show_flag:
            cap = cv.VideoCapture(video_path)
            for i in range(len(boxes)):
                ret, frame = cap.read()
                if ret is False:
                    break
                Tracking.__showBndbox(frame, boxes[i], color)
            cap.release()
            cv.destroyAllWindows()

        if save_flag:
            for i in range(len(boxes)):
                xmladdress = save_path + "/" + str(i + 1) + ".xml"
                XMLAnalyse.writeXML(xmladdress, boxes[i])

        return boxes

    @staticmethod
    def __settle_feature_points(bndbox_size, frame):
        xmin = bndbox_size[0]
        xmax = bndbox_size[1]
        ymin = bndbox_size[2]
        ymax = bndbox_size[3]

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        features = cv.goodFeaturesToTrack(frame_gray, mask=None, **Tracking.feature_params, useHarrisDetector=False,
                                          k=0.04)
        features = features.reshape(-1, 2)
        points = []
        for point in features:
            if (point[0] >= xmin) & (point[0] <= xmax) & (point[1] >= ymin) & (point[1] <= ymax):
                points.append(point)
        return np.array(points).reshape(-1, 1, 2)

    @staticmethod
    def __cal_box_size_list(folder_path, xmllist):
        sizelist = []
        frame_index = 0
        for i in range(len(xmllist) - 1):
            bndbox1 = XMLAnalyse.analyseXML(folder_path + '/' + xmllist[i])
            bndbox2 = XMLAnalyse.analyseXML(folder_path + '/' + xmllist[i + 1])
            bndbox = bndbox1.copy()
            frameinterval = int(xmllist[i + 1][0:-4]) - int(xmllist[i][0:-4])
            changes = [(bndbox2[k] - bndbox1[k]) / frameinterval for k in range(len(bndbox1))]
            sizelist.append([bndbox[1] - bndbox[0], bndbox[3] - bndbox[2]])
            frame_index = frame_index + 1
            for j in range(frameinterval - 1):
                bndbox = [bndbox[k] + changes[k] for k in range(len(bndbox))]
                sizelist.append([bndbox[1] - bndbox[0], bndbox[3] - bndbox[2]])
                frame_index = frame_index + 1
        sizelist.append([bndbox2[1] - bndbox2[0], bndbox2[3] - bndbox2[2]])
        return sizelist

    @staticmethod
    def __showBndbox(frame, bndbox, color):
        cv.line(frame, (int(bndbox[0]), int(bndbox[2])), (int(bndbox[0]), int(bndbox[3])), color, 2)  # left
        cv.line(frame, (int(bndbox[1]), int(bndbox[2])), (int(bndbox[1]), int(bndbox[3])), color, 2)  # right
        cv.line(frame, (int(bndbox[0]), int(bndbox[2])), (int(bndbox[1]), int(bndbox[2])), color, 2)  # top
        cv.line(frame, (int(bndbox[0]), int(bndbox[3])), (int(bndbox[1]), int(bndbox[3])), color, 2)  # bottom
        cv.imshow('frame', frame)
        cv.waitKey(1)
