import cv2
from lp_image_detect import recognize_lp_from_image

# setting video in and out path
video_path = 'Data/video/video_2023-04-13_18-55-23.mp4'
out_video_path = 'Outputs/detections/detect_video_2023-04-13_18-55-23.mp4'

# read video
try:
    vid = cv2.VideoCapture(int(video_path))
except:
    vid = cv2.VideoCapture(video_path)

# setting out video
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(vid.get(cv2.CAP_PROP_FPS))
codec = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(out_video_path, codec, fps, (width, height))

# read video and detect license plate
frame_num = 0
done_task = True
while done_task:
    return_value, image = vid.read()
    try:
        image = recognize_lp_from_image(image)
        out.write(image)
    except:
        done_task = False
