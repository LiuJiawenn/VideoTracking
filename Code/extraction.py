import cv2
import os
import numpy

class Extract:
    # these three variable need to be set manually###################################################################
    # folder_path: the directory where stores the videos to extract (one or more than one all ok)
    # save_path: a folder where you want to save the frames this code will create different folder automatically for
    #            different video
    # frame_interval: Interval of extracted frames
    #################################################################################################################
    @staticmethod
    def extract_interval_fixed(folder_path='../video data', save_path='../extracted images/interval 30',
                               frame_interval=30):
        videos = os.listdir(folder_path)  # find all videos in this folder
        videos.sort(key=lambda x: int(x[7:-4]))  # range the videos in number order

        for video in videos:
            video_path = folder_path + '/' + video
            frames_path = save_path + '/' + video[:-4]  # Each video has its own keyframe storage path
            if not os.path.exists(frames_path):  # create directory
                os.makedirs(frames_path)

            count = 0
            count_ectracted = 0

            vc = cv2.VideoCapture(video_path)  # import video file
            print("video " + video[:-4])
            # The first frame must be taken out
            # It is impossible to take out the 1st, 30th, 60th.... sheets at the same time
            # by just taking one remainder
            ret, frame = vc.read()
            count = count + 1
            if ret:
                count_ectracted = count_ectracted + 1
                image_path = frames_path + '/{}.jpg'.format(str(count))
                cv2.imwrite(image_path, frame)
            else:
                print("video open failed")
                vc.release()
                break

            # Remained frames
            ret, frame = vc.read()
            while ret:
                count = count + 1
                temp_frame = frame
                if count % frame_interval == 0:
                    count_ectracted = count_ectracted + 1
                    image_path = frames_path + '/{}.jpg'.format(str(count))
                    cv2.imwrite(image_path, frame)

                ret, frame = vc.read()

            # the last frame
            if not count % frame_interval == 0:
                count_ectracted = count_ectracted + 1
                image_path = frames_path + '/{}.jpg'.format(str(count))
                cv2.imwrite(image_path, temp_frame)
            print(str(count_ectracted) + " frames extracted")
            vc.release()

    # four variable need to be set manually ###############################################################
    # folder_path: the directory where stores the videos to extract (one or more than one all ok)
    # save_path: a folder where you want to save the frames this code will create different folder for different videos
    # frame_number: number of frames you need to extract from each video
    # flag: if you want Euclid difference, make flag = 1
    #       if you want Optical Flow difference, make flag = 2
    #######################################################################################################
    @staticmethod
    def extract_difference_fixed(folder_path='../video data', save_path='../extracted images/Euclid 20 frames',
                                 frame_number=20, flag=1, extract_flag=0):
        videos = os.listdir(folder_path)  # Find all the videos under the folder
        videos.sort(key=lambda x: int(x[7:-4]))  # Videos are sorted by number order
        key_frames_list = []

        for video in videos:
            video_path = folder_path + '/' + video

            list_difference = []  # to record the difference between frames
            count = 0  # to count frames
            count_extracted = 0
            vc = cv2.VideoCapture(video_path)  # import video file
            ret, frame_1 = vc.read()
            count = count + 1  # The first frame is definitely to be extracted
            count_extracted = count_extracted + 1
            key_frames = [1]

            ret, frame_2 = vc.read()
            while ret:
                count = count + 1
                if flag == 1:
                    list_difference.append(Extract.__eucliddiff(frame_1, frame_2))
                else:
                    list_difference.append(Extract.__opticalFlowdiff(frame_1, frame_2))
                frame_1 = frame_2
                ret, frame_2 = vc.read()

            interval = sum(list_difference) / (frame_number - 1)

            sum_difference = 0
            for i in range(count-1):
                sum_difference = sum_difference + list_difference[i]
                if sum_difference >= interval:
                    key_frames.append(i + 1)
                    sum_difference = sum_difference - interval
                    count_extracted = count_extracted + 1

                if count_extracted == (frame_number - 1):
                    break

            key_frames.append(count)
            if extract_flag:
                frames_path = save_path + '/' + video[:-4]  # Each video has its own keyframe storage path
                if not os.path.exists(frames_path):
                    os.makedirs(frames_path)

                for i in key_frames:
                    vc.set(cv2.CAP_PROP_POS_FRAMES, i-1)
                    ret, frame = vc.read()
                    image_path = frames_path + '/{}.jpg'.format(str(i))
                    cv2.imwrite(image_path, frame)

            vc.release()
            key_frames_list.append(key_frames.copy())
        return key_frames_list

    @staticmethod
    def __caldist(a, b, c, d):
        return abs(a - c) + abs(b - d)

    @staticmethod
    def __eucliddiff(frame1, frame2):
        return numpy.linalg.norm(frame1 - frame2)

    @staticmethod
    def __opticalFlowdiff(frame1, frame2):
        # Edge point detection parameters   maxcorners:Number of key points detected, 0 means no upper limit
        #                                   minDistance: Minimum distance between key points is 7 pixels
        feature_params = dict(maxCorners=0, qualityLevel=0.1, minDistance=7, blockSize=7)
        # KLT Optical flow parameters
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.02))

        old_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params, useHarrisDetector=False, k=0.04)
        # p0: List of coordinates for storing key points

        frame_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Selection according to the state
        # st==1 status of 1 means that the key point of the previous frame is found in the second frame
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Delete stationary points
        k = 0
        for i, (new0, old0) in enumerate(zip(good_new, good_old)):
            a0, b0 = new0.ravel()
            c0, d0 = old0.ravel()
            dist = Extract.__caldist(a0, b0, c0, d0)
            if dist > 0.5:
                good_new[k] = good_new[i]
                good_old[k] = good_old[i]
                k = k + 1

        good_new = good_new[:k]
        good_old = good_old[:k]
        # Return the mean square error of these two point sets
        return numpy.linalg.norm(good_old - good_new)
