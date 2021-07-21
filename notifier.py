# import inotify.adapters
import os
import glob
from main import AppleDetector
import time


image_list = glob.glob('test/*')
extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
apple_detector = AppleDetector()
print("#"*50)
print("Add Image to test folder")
while True:
    for each in glob.glob('test/*'):
        if each not in image_list:
            if os.path.splitext(each)[1] in extensions:
                filename = os.path.basename(each)
                time.sleep(2)
                try:
                    apple_detector.get_format(each, filename)
                except:
                    print("Please add a file of small size")
                    continue

            else:
                print("Please enter an image file")
            image_list.append(each)

# for event in notifier.event_gen():
#     if event is not None:
#         print(event)      # uncomment to see all events generated
#         if 'IN_CREATE' in event[1]:

#              print (f"file {event[3]} Added in  {event[2]}")
#              image_path = event[2] + os.sep + event[3]
#              apple_detector.get_format(image_path, event[3])