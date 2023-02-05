from test import AccuracyTest
from tracking import Tracking
import os
import numpy as np

# Example 1 :Call tracking algorithme one by one
# true_path = '../label manual/labeled all frames/s8_r5_a16'
# algo_order = ['linear', 'opticalFlow', 'bidirection', 'csrt', 'medianflow', 'mosse', 'mil', 'boosting', 'kcf', 'tld']
#
# for algo in algo_order:
#     folder_path = '../label manual/interval fixed/interval 25'
#     save_path = '../result/interval fixed/' + algo + '/interval 25'
#     # tracking
#     Tracking.callalgobyname(folder_path=folder_path, save_path=save_path, algo=algo)
#     # test
#     re = AccuracyTest.accuracy_test(predict_path=save_path,
#                                     true_path=true_path,
#                                     manu_path=folder_path)


# Example 2 :Call one tracking algorithme with different key frame interval
data = np.zeros((7, 14))
true_path = '../label manual/labeled all frames/s8_r5_a16'
col = 0
for interval in [5]:#[5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100, 200, 1000]:
    data[0, col] = interval
    folder_path = '../label manual/interval fixed/interval ' + str(interval)
    save_path = '../result/intervel fixed/opticalFlow/interval ' + str(interval)
    if not os.path.exists(save_path):  # create directory
        os.makedirs(save_path)
    Tracking.linear_tracking(folder_path=folder_path,
                             save_path=save_path)
    # Tracking.optical_flow_tracking(folder_path=folder_path,
    #                                save_path=save_path)

    # Tracking.bidirection_optical_flow_tracking(folder_path=folder_path,
    #                          save_path=save_path)
    # Tracking.openCV_tracking(folder_path=folder_path,
    #                          algo_flag='mosse',
    #                          save_path=save_path)

    re = AccuracyTest.accuracy_test(predict_path=save_path,
                                    true_path=true_path,
                                    manu_path=folder_path)
    data[1:7, col] = re
    col = col + 1
    print("\n")

np.savetxt("C:/Users/seren/Desktop/bi.csv", data, delimiter=',')
