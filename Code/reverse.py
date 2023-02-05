import shutil
import cv2 as cv
import os


# reverse the video at video_path, store the reverse video at output_path
def reverse_a_video(video_path='../video data/s8_r5_a16.avi',
                    output_path='../video data/reverse_s8_r5_a16.avi'):
    cap = cv.VideoCapture(video_path)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(output_path, fourcc, 30.0, (1520, 960))
    frames = []
    ok, frame = cap.read()

    while ok:
        frames.append(frame)
        ok, frame = cap.read()

    print(len(frames))
    frames.reverse()

    for frame in frames:
        out.write(frame)

    cap.release()
    cv.destroyAllWindows()


# The annotation file is renamed in reverse order,
# for example, the video has 689 frames, so 2.xml should become 688.xml
def reverse_annotation_files(original_path='../label manual/labeled all frames/s8_r5_a16',
                             save_folder='../label manual/reverse',
                             interval_list=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100, 200, 300, 1000]):
    original_list = os.listdir(original_path)  # Find all the video annotation files under the folder
    original_list.sort(key=lambda x: int(x[0:-4]))  # Files are sorted in numerical order

    frame_number = len(original_list)
    for interval in interval_list:
        save_path = save_folder + '/interval ' + str(interval)
        if not os.path.exists(save_path):  # create directory
            os.makedirs(save_path)
        # copy the first frame
        filePath = os.path.join(original_path, original_list[0])
        new = str(frame_number + 1 - int(original_list[0][0:-4])) + '.xml'
        save = os.path.join(save_path, new)
        shutil.copy(filePath, save)
        # copy the last frame
        filePath = os.path.join(original_path, original_list[-1])
        new = str(frame_number + 1 - int(original_list[-1][0:-4])) + '.xml'
        save = os.path.join(save_path, new)
        shutil.copy(filePath, save)
        # copy the rest
        for i in range(len(original_list)):
            filePath = os.path.join(original_path, original_list[i])
            if (i + 1) % interval == 0:
                new = str(frame_number + 1 - int(original_list[i][0:-4])) + '.xml'
                save = os.path.join(save_path, new)
                shutil.copy(filePath, save)


