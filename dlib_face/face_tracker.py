import dlib
import cv2
'''
使用 dlib 的相关滤波器跟踪器进行实时目标跟踪

1、初始化了一个 dlib.correlation_tracker() 对象。
2、打开摄像头并读取第一帧。
3、使用 cv2.selectROI() 函数选择要跟踪的目标区域（可以通过鼠标交互选择）。
4、使用 tracker.start_track() 函数启动跟踪器，传入第一帧和目标区域的矩形框。
5、在每一帧中，使用 tracker.update() 函数更新跟踪器，获取目标的新位置。
6、解析跟踪结果并在图像上绘制矩形框。
7、显示结果图像，并按下 'q' 键退出循环。
'''

# 初始化相关滤波器跟踪器
tracker = dlib.correlation_tracker()

# 打开摄像头
cap = cv2.VideoCapture(0)

# 检查摄像头是否打开成功
if not cap.isOpened():
    print("无法打开摄像头")
    exit()
else:
    print('FPS:', cap.get(cv2.CAP_PROP_FPS),
          ', Width:', cap.get(cv2.CAP_PROP_FRAME_WIDTH),
          ', Height:', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 读取第一帧并选择跟踪目标
ret, frame = cap.read()
if not ret:
    print("无法读取帧")
    exit()

# 选择跟踪目标的矩形区域（可以使用鼠标交互选择）
bbox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)

# 启动跟踪器
tracker.start_track(frame, dlib.rectangle(bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))

while True:
    # 读取新帧
    ret, frame = cap.read()
    if not ret:
        print("无法读取帧")
        break

    # 更新跟踪器
    tracker.update(frame)

    # 获取跟踪结果
    pos = tracker.get_position()

    # 解析位置信息
    startX = int(pos.left())
    startY = int(pos.top())
    endX = int(pos.right())
    endY = int(pos.bottom())

    # 绘制跟踪结果
    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # 显示结果
    cv2.imshow("Frame", frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()