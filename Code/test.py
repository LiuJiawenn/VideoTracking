from calcul_overlapping import CalculOverLapping as cl
from xml_analyse import XMLAnalyse
import os


class AccuracyTest:
    @staticmethod
    def cvat_accuracy_test(predict_path="../label manual/CVAT export/CVAT siamMask.xml",
                           true_path="../label manual/labeled all frames/s8_r5_a16",
                           manu_path="../label manual/labeled key frames20/s8_r5_a16"):
    # We should determine if the contents of the two folders are the same length
    # but i didn't
        predict_bndbox = XMLAnalyse.analyseCVATXML(predict_path)

        true_list = os.listdir(true_path)  # Find all the video annotation files under the folder
        true_list.sort(key=lambda x: int(x[0:-4]))  # Sort by numerical order

        manu_list = os.listdir(manu_path)
        manu_list.sort(key=lambda x: int(x[0:-4]))

        bad_counter = 0
        good_counter = 0
        sum_percent = 0
        labeled_counter = 0

        for i in range(len(predict_bndbox)):
            if int(manu_list[labeled_counter][0:-4]) == (i + 1):
                if labeled_counter < len(manu_list) - 1:
                    labeled_counter = labeled_counter + 1
                continue

            bndbox = predict_bndbox[i]
            true_box = XMLAnalyse.analyseXML(true_path + '/' + true_list[i])
            overlapped = cl.calareaoverlapped(true_box, bndbox)
            if overlapped == 0:
                bad_counter = bad_counter + 1
            else:
                good_counter = good_counter + 1
                sum_percent = sum_percent + overlapped

        if good_counter == 0:
            sum_percent = 0
        else:
            sum_percent = sum_percent / good_counter

        print(f"There are {len(predict_bndbox)} frames in this video")
        print(f"{labeled_counter + 1} frames are labeled by hand")
        print(f"{bad_counter} frames are bad tracked")
        print(f"{good_counter} frames are tracked and the accuracy is {sum_percent}")

    @staticmethod
    def accuracy_test(predict_path="../track_result/s8_r5_a16_optiacl_flow",
                      true_path="../label manual/labeled all frames/s8_r5_a16",
                      manu_path="../label manual/labeled key frames20/s8_r5_a16"):

        predict_list = os.listdir(predict_path)
        predict_list.sort(key=lambda x: int(x[0:-4]))

        true_list = os.listdir(true_path)
        true_list.sort(key=lambda x: int(x[0:-4]))

        manu_list = os.listdir(manu_path)
        manu_list.sort(key=lambda x: int(x[0:-4]))

        bad_counter = 0
        good_counter = 0
        sum_percent = 0
        labeled_counter = 0

        for i in range(len(predict_list)):
            if int(manu_list[labeled_counter][0:-4]) == (i + 1):
                if labeled_counter < len(manu_list) - 1:
                    labeled_counter = labeled_counter + 1
                continue

            predict_box = XMLAnalyse.analyseXML(predict_path + '/' + predict_list[i])
            true_box = XMLAnalyse.analyseXML(true_path + '/' + true_list[i])
            overlapped = cl.calareaoverlapped(true_box, predict_box)
            if overlapped == 0:
                bad_counter = bad_counter + 1
            else:
                good_counter = good_counter + 1
                sum_percent = sum_percent + overlapped

        if good_counter == 0:
            sum_percent = 0
            trackable = 0
            total_percent = 0
        else:
            total_percent = sum_percent / (good_counter + bad_counter)
            sum_percent = sum_percent / good_counter
            trackable = good_counter/(len(predict_list)-labeled_counter - 1)

        print(f"There are {len(predict_list)} frames in this video")
        print(f"{labeled_counter + 1} frames are labeled by hand")
        print(f"{bad_counter} frames are bad tracked")
        print(f"{good_counter} frames are tracked")
        print(f"If we only see the tracked frames, the accuracy is {sum_percent}")
        print(f"If we count all frame, the accuracy is {total_percent} ")
        print(f"trackable:{trackable}")
        return [trackable, sum_percent, total_percent, good_counter, bad_counter, (labeled_counter + 1)]
