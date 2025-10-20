'''
moivepy / cv2 / librosa / ffmpeg
变速涉及对时间轴的调整，改变speed而不改变音调(audio)或帧率(video)
'''
import cv2
import librosa
import soundfile as sf
from moviepy.editor import VideoFileClip, AudioFileClip, vfx
import numpy as np


audio_path = r'D:\LipData\CMLR\audio\s1\20170121\section_1_000.80_002.91.wav'
video_path = r'D:\LipData\CMLR\video\s1\20170121\section_1_000.80_002.91.mp4'
speedx = 0.8
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(fps, num_frame)
cap.release()
#
# y, sr = librosa.load(audio_path, sr=None)
# sy = librosa.effects.time_stretch(y, speedx)  # 不改采样率
# sf.write('audio_speed.wav', sy, sr)
#
#
video = VideoFileClip(video_path)
new_video = video.fx(vfx.speedx, speedx)   # 不改帧率，改变帧数
new_video.write_videofile('video_speed.mp4')
cap = cv2.VideoCapture('video_speed.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(fps, num_frame)
cap.release()

new_video = video.set_duration(video.duration / speedx).set_fps(video.fps * speedx)   # 改变帧率，不改变帧数
new_video.write_videofile('video_speed2.mp4')
cap = cv2.VideoCapture('video_speed2.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(fps, num_frame)
cap.release()



# 视频参数
width = 640  # 视频宽度
height = 480  # 视频高度
fps = 30      # 原始帧率
speed_factor = 1.5  # 变速因子

input_file = open('video.yuv', 'rb')
output_file = open('video_fast.yuv', 'wb')

frame_size = width * height * 3 // 2  # YUV420P 格式

frames = []
while True:
    raw = input_file.read(frame_size)
    if not raw:
        break
    yuv = np.frombuffer(raw, dtype=np.uint8).reshape((height * 3 // 2, width))
    frames.append(yuv)

# 调整帧速率
step = max(1, int(1 / speed_factor))

# 写入调整后的帧
# 加速：跳过帧；减速：复制帧或插帧
for i in range(0, len(frames), step):
    output_file.write(frames[i].tobytes())

input_file.close()
output_file.close()

