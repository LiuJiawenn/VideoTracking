from test import AccuracyTest
import numpy as np

# ndarray for saving result in csv format
data = np.zeros((14, 2))
count = 0

# read annotation file according to key frame interval
for interval in [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100, 200, 1000]:
    folder_path1 = '../result/interval fixed/csrt/interval ' + str(interval)
    folder_path2 = '../result/reverse/csrt/interval ' + str(interval)

    manu_path = '../label manual/interval fixed/interval ' + str(interval)

    result1 = AccuracyTest.accuracy_test(predict_path=folder_path1,
                                         true_path='../label manual/labeled all frames/s8_r5_a16',
                                         manu_path= manu_path)

    result2 = AccuracyTest.accuracy_test(predict_path=folder_path2,
                                         true_path='../label manual/labeled all frames/s8_r5_a16',
                                         manu_path=manu_path)
    # fill the accuracy in ndarray
    data[count, 0] = result1[2]
    data[count, 1] = result2[2]
    count = count + 1

# save test result
np.savetxt("C:/Users/seren/Desktop/reserve.csv", data, delimiter=',')
