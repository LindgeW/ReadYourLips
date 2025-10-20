import dlib
import cv2
import numpy as np

'''
人脸对齐（Face Alignment）是指将检测到的人脸关键点进行标准化处理，使得不同姿态、表情和光照条件下的人脸能够对齐到一个标准化的坐标系中。
人脸对齐在人脸识别、表情分析等领域有广泛应用。
基于dlib检测到的人脸关键点，可以使用仿射变换（Affine Transformation，包括旋转、缩放、平移）来实现人脸对齐。
'''


def face_alignment():
    # 加载dlib的预训练人脸检测器和关键点预测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r"D:\Codes\shape_predictor_68_face_landmarks.dat")

    # 打开视频文件
    video_path = r"../lip_roi_proc/me.mp4"
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}.")
        exit()
    else:
        print('FPS:', cap.get(cv2.CAP_PROP_FPS),
              ', Width:', cap.get(cv2.CAP_PROP_FRAME_WIDTH),
              ', Height:', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        # 读取视频帧
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # 将帧转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        faces = detector(gray, 2)  # 原图放大2倍检测人脸

        # 遍历检测到的人脸
        for face in faces:
            # 获取人脸关键点
            landmarks = predictor(gray, face)

            # 提取左眼和右眼的关键点
            # left_eye = (landmarks.part(36).x, landmarks.part(36).y)
            # right_eye = (landmarks.part(45).x, landmarks.part(45).y)
            left_eye = (np.mean([landmarks.part(i).x for i in range(36, 42)]),
                        np.mean([landmarks.part(i).y for i in range(36, 42)]))
            right_eye = (np.mean([landmarks.part(i).x for i in range(42, 48)]),
                         np.mean([landmarks.part(i).y for i in range(42, 48)]))
            # 计算眼睛中心点
            eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)

            # 计算眼睛之间的角度
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            #angle = np.degrees(np.arctan2(dY, dX))
            angle = np.arctan(dY / dX) * 180.0 / np.pi
            # 计算旋转矩阵
            rot_matrix = cv2.getRotationMatrix2D(eye_center, angle, 1.0)

            # 进行仿射变换
            aligned_face = cv2.warpAffine(frame, rot_matrix, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_CUBIC)

            # 显示对齐后的脸
            cv2.imshow("Aligned Face", aligned_face)

        # 显示原始帧
        cv2.imshow("Original Frame", frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(30) & 0xFF == ord('q'):  # 延时30ms
            break

    # 释放视频对象并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()


def improved_face_alignment():
    FACE_SIZE_REF = (400, 400)  # 对齐的人脸图像大小
    LEFT_EYE_REF = (0.3, 0.3)   # 对齐的人脸图像中左眼中心点位置
    RIGHT_EYE_REF = (0.7, 0.7)  # 对齐的人脸图像中右眼中心点位置
    # 对齐的人脸左右眼中心点坐标
    left_eye_ref = (LEFT_EYE_REF[0] * FACE_SIZE_REF[0], LEFT_EYE_REF[1] * FACE_SIZE_REF[1])
    right_eye_ref = (RIGHT_EYE_REF[0] * FACE_SIZE_REF[0], RIGHT_EYE_REF[1] * FACE_SIZE_REF[1])
    # 期望的左右眼中心距离
    dist_ref = right_eye_ref[0] - left_eye_ref[0]

    # 加载dlib的预训练人脸检测器和关键点预测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r"D:\Codes\shape_predictor_68_face_landmarks.dat")

    # 打开视频文件
    # video_path = r"me.mp4"
    video_path = r"D:\LipData\CMLR\video\s5\20090728\section_6_001.69_005.57.mp4"
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}.")
        exit()
    else:
        print('FPS:', cap.get(cv2.CAP_PROP_FPS),
              ', Width:', cap.get(cv2.CAP_PROP_FRAME_WIDTH),
              ', Height:', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    i = 1
    while True:
        # 读取视频帧
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # 将帧转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        faces = detector(gray)

        # 遍历检测到的人脸
        for face in faces:
            # 获取人脸关键点
            landmarks = predictor(gray, face)

            # 提取左眼和右眼的关键点
            # left_eye = (landmarks.part(36).x, landmarks.part(36).y)
            # right_eye = (landmarks.part(45).x, landmarks.part(45).y)
            left_eye = (np.mean([landmarks.part(i).x for i in range(36, 42)]),
                        np.mean([landmarks.part(i).y for i in range(36, 42)]))
            right_eye = (np.mean([landmarks.part(i).x for i in range(42, 48)]),
                         np.mean([landmarks.part(i).y for i in range(42, 48)]))
            # 检测出的左右眼中心点 (旋转中心)
            eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)

            # 计算眼睛之间的角度
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            #angle = np.degrees(np.arctan2(dY, dX))
            angle = np.arctan(dY / dX) * 180.0 / np.pi  # 旋转角度
            scale = dist_ref / np.sqrt(dX**2 + dY**2)   # 各项同性缩放因子 (期望的左右眼中心距离除以检测的左右眼中心距离)
            # 计算旋转矩阵
            M = cv2.getRotationMatrix2D(eye_center, angle, scale)
            # 计算x, y方向平移量
            M[0, 2] += FACE_SIZE_REF[0] / 2 - eye_center[0]
            M[1, 2] += left_eye_ref[1] - eye_center[1]

            # 进行仿射变换
            aligned_face = cv2.warpAffine(frame, M, FACE_SIZE_REF, flags=cv2.INTER_CUBIC)

            # 显示对齐后的脸
            cv2.imshow("Aligned Face", aligned_face)
            i += 1
            cv2.imwrite('{}.jpg'.format(i), aligned_face)

        # 显示原始帧
        cv2.imshow("Original Frame", frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(10) & 0xFF == ord('q'):  # 延时10ms
            break

    # 释放视频对象并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()


#   ----------------------------------------------------------------------------------------------

def face_alignment2():
    # 加载dlib的预训练人脸检测器和关键点预测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r"D:\Codes\shape_predictor_68_face_landmarks.dat")

    # 打开视频文件
    video_path = r"D:\LipData\CMLR\video\s5\20090728\section_6_001.69_005.57.mp4"

    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}.")
        exit()
    else:
        print('FPS:', cap.get(cv2.CAP_PROP_FPS),
              ', Width:', cap.get(cv2.CAP_PROP_FRAME_WIDTH),
              ', Height:', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    i = 0
    while True:
        # 读取视频帧
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # 将帧转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        faces = detector(gray, 2)  # 放大原图2倍进行检测

        # 遍历检测到的人脸
        for face in faces:
            # 获取人脸关键点
            landmarks = predictor(gray, face)

            # x, y, w, h = face.left(), face.top(), face.width(), face.height()
            # cropped_face = frame[y:y + h, x:x + w]
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cropped_face = frame[y1: y2, x1: x2]

            # 提取左眼角和右眼角的关键点
            # left_eye = (landmarks.part(36).x, landmarks.part(36).y)
            # right_eye = (landmarks.part(45).x, landmarks.part(45).y)
            left_eye = (np.mean([landmarks.part(i).x for i in range(36, 42)]),
                        np.mean([landmarks.part(i).y for i in range(36, 42)]))
            right_eye = (np.mean([landmarks.part(i).x for i in range(42, 48)]),
                         np.mean([landmarks.part(i).y for i in range(42, 48)]))
            # 计算眼睛之间的角度
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            #angle = np.degrees(np.arctan2(dY, dX))
            angle = np.arctan(dY / dX) * 180.0 / np.pi
            # 计算旋转矩阵
            M = cv2.getRotationMatrix2D((cropped_face.shape[1] / 2, cropped_face.shape[0] / 2), angle, 1.0)

            # 进行仿射变换
            aligned_face = cv2.warpAffine(cropped_face, M, (cropped_face.shape[1], cropped_face.shape[0]))

            aligned_face = cv2.resize(aligned_face, (180, 180))

            # 显示对齐后的脸
            cv2.imshow("Aligned Face", aligned_face)
            i += 1
            cv2.imwrite('{}.jpg'.format(i), aligned_face)

        # 显示原始帧
        cv2.imshow("Original Frame", frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(30) & 0xFF == ord('q'):  # 延时30ms
            break

    # 释放视频对象并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()



# import imutils
# from imutils.face_utils import rect_to_bb, FaceAligner
from imutil_face_align import rect_to_bb, FaceAligner
def face_alignment3():
    # 加载dlib的预训练人脸检测器和关键点预测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r"D:\Codes\shape_predictor_68_face_landmarks.dat")
    fa = FaceAligner(predictor, desiredLeftEye=(0.3, 0.3), desiredFaceWidth=256)

    # 打开视频文件
    video_path = r"D:\LipData\CMLR\video\s5\20090728\section_6_001.69_005.57.mp4"
    # video_path = 'me.mp4'
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}.")
        exit()
    else:
        print('FPS:', cap.get(cv2.CAP_PROP_FPS),
              ', Width:', cap.get(cv2.CAP_PROP_FRAME_WIDTH),
              ', Height:', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    i = 0
    while True:
        # 读取视频帧
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # 将帧转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        faces = detector(gray, 1)  # 2: 放大原图2倍进行检测

        # 遍历检测到的人脸
        for face in faces:
            x, y, w, h = rect_to_bb(face)
            aligned_face = fa.align(frame, gray, face)
            # 显示对齐后的脸
            cv2.imshow("Aligned Face", aligned_face)
            i += 1
            cv2.imwrite('{}.jpg'.format(i), aligned_face)

        # 显示原始帧
        cv2.imshow("Original Frame", frame)

        # 按下 'q' 键退出循环
        if cv2.waitKey(30) & 0xFF == ord('q'):  # 延时30ms
            break

    # 释放视频对象并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()


# improved_face_alignment()
face_alignment3()