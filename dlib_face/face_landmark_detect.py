import dlib
import cv2

# 加载dlib的预训练人脸检测器和关键点预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"D:\Codes\shape_predictor_68_face_landmarks.dat")

# 打开视频文件或摄像头
cap = cv2.VideoCapture(0)  # 0表示打开默认摄像头，也可以传入视频文件路径

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
else:
    print('FPS:', cap.get(cv2.CAP_PROP_FPS),
          ', Width:', cap.get(cv2.CAP_PROP_FRAME_WIDTH),
          ', Height:', cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
          ', FrameNum:', cap.get(cv2.CAP_PROP_FRAME_COUNT),
          ', Duration:', cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))

while True:
    # 读取视频帧
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # 将帧转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = detector(gray)

    # 遍历检测到的人脸
    for face in faces:
        # 获取人脸关键点
        landmarks = predictor(gray, face)

        # 绘制关键点
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # 显示结果帧
    cv2.imshow("Face Landmarks", frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频对象并关闭所有窗口
cap.release()
cv2.destroyAllWindows()