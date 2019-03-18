import cv2
import os

# seedNum = 12
# directory = './all_images/'
images = os.listdir(directory)

# video_dir = './videos/'
# nn = seedNum + 5
# video_name = video_dir+'video%d.mp4' %nn
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video = cv2.VideoWriter(video_name, fourcc, 96.0, (512, 512))

# y = 607
# noise_seed = 1
# t = 20

for i in range(len(images)):
    file = 'cat_%d__noise_%d__trunc_%d.jpeg' %(y, noise_seed, t)
    name = directory+file
    if (os.path.isfile(name)):
        video.write(cv2.imread(name))
    t += 1

for i in range(len(images)):
    t -= 1
    file = 'cat_%d__noise_%d__trunc_%d.jpeg' %(y, noise_seed, t)
    name = directory+file
    if (os.path.isfile(name)):
        video.write(cv2.imread(name))



cv2.destroyAllWindows()
video.release()
