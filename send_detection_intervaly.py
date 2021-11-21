import time
import collections
import soracom
import json

INTERVAL = 60

classes = []

with open('classes.txt') as f:
    classes = f.readlines()
    classes = [cls.rstrip('\n') for cls in classes]

while True:
    time.sleep(INTERVAL)
    with open('detections.txt') as f:
        lines = f.readlines()
        # print(lines)
        timestamps = []
        detections = []
        length = len(lines)
        print(length)
        for line in lines:
            timestamps.append(float(line.split(' ')[0]))
            detections.append(eval(''.join(line.split(' ')[1:])))

        current = time.time()
        for timestamp in timestamps:
            if timestamp > (current - 60):
                startIndex = timestamps.index(timestamp)
                break
        
        itemCount = []

        data_count = length - startIndex
        print(data_count)
        calibrate = INTERVAL / data_count

        for i in range(startIndex, length):            
            for detection in detections[i]:
                itemCount.append(detection[0])
                # print(itemCount)

        counter = collections.Counter(itemCount)

        data_dic = {}
        for cls in classes:
            data_dic[cls] = int(counter[cls] * calibrate)
        print(data_dic)
        
    soracom.send_data_to_endpoint(json.dumps(data_dic))
    