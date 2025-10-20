import cv2
import dlib
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import glob
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"D:\Codes\shape_predictor_68_face_landmarks.dat")
W, H = 128, 64

def get_position(desired_size, padding=0.25):
    # Average positions of face points 17-67  (reference shape)
    x = [0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
         0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
         0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
         0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
         0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
         0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
         0.553364, 0.490127, 0.42689]  # mean_face_shape_x

    y = [0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
         0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
         0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
         0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
         0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
         0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
         0.784792, 0.824182, 0.831803, 0.824182]  # mean_face_shape_y

    x, y = np.array(x), np.array(y)
    x = (x + padding) / (2 * padding + 1) * desired_size
    y = (y + padding) / (2 * padding + 1) * desired_size
    return np.array(list(zip(x, y)))


def transformation_from_points(points1, points2):
    # points1：需要对齐的人脸关键点
    # points2：对齐的模板人脸(平均脸关键点)
    '''0 - 先确定是float数据类型 '''
    points1 = np.copy(points1).astype(np.float64)
    points2 = np.copy(points2).astype(np.float64)
    '''1 - 消除平移的影响 '''
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    '''2 - 消除缩放的影响 '''
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    '''3 - 计算矩阵M=BA^T；对矩阵M进行SVD分解；计算得到R '''
    # ||RA-B||; M=BA^T
    A = points1.T  # 2xN
    B = points2.T  # 2xN
    M = np.dot(B, A.T)
    U, S, Vt = np.linalg.svd(M)
    R = np.dot(U, Vt)
    '''4 - 构建仿射变换矩阵 '''
    s = s2 / s1
    sR = s * R
    c1 = c1.reshape(2, 1)
    c2 = c2.reshape(2, 1)
    T = c2 - np.dot(sR, c1)  # 模板人脸的中心位置减去需要对齐的中心位置（经过旋转和缩放之后）
    trans_mat = np.hstack([sR, T])  # 2x3
    return trans_mat


# def transformation_from_points(points1, points2):
#     # 注：points1和points2是np.matrix类型，*就相当于矩阵乘，即np.array的dot()和@
#     points1 = points1.astype(np.float64)
#     points2 = points2.astype(np.float64)
#     c1 = np.mean(points1, axis=0)
#     c2 = np.mean(points2, axis=0)
#     points1 -= c1
#     points2 -= c2
#     s1 = np.std(points1)
#     s2 = np.std(points2)
#     points1 /= s1
#     points2 /= s2
#     U, S, Vt = np.linalg.svd(points1.T * points2)
#     R = (U * Vt).T
#     return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)),
#                       np.matrix([0., 0., 1.])])     # 3x3


def get_landmarks(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)  # 默认传入的图像为灰度图
    if len(faces) == 0:
        return None
    shape = predictor(gray, faces[0])
    lms = np.asarray([(p.x, p.y) for p in shape.parts()])
    return lms


def load_video(path):
    cap = cv2.VideoCapture(path)
    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"Error: Could not open video at {path}.")
        exit()
    else:
        # print('FPS:', cap.get(cv2.CAP_PROP_FPS),
        #       ', Width:', cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        #       ', Height:', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        pass
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def align_img(img_dir, save_dir, desired_size=256):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 加载目录中的图像序列
    files = list(os.listdir(img_dir))
    files = [file for file in files if file.find('.jpg') != -1]
    shapes = []
    imgs = []
    for file in files:
        I = cv2.imread(os.path.join(img_dir, file), 0)
        shape = get_landmarks(I)
        if shape is None:
            print(file)
            continue
        imgs.append(I)
        shapes.append(shape[17:])  # 不包括脸部轮廓

    mean_shape = get_position(desired_size)  # 模板脸(平均脸)

    for i, shape in enumerate(shapes):
        M = transformation_from_points(np.matrix(shape), np.matrix(mean_shape))
        img = cv2.warpAffine(imgs[i],
                             M[:2],  # [R|t]  2 x 3 仿射变换矩阵
                             dsize=(desired_size, desired_size),  # output size: (cols, rows)
                             borderMode=cv2.BORDER_TRANSPARENT)
        cv2.imwrite(os.path.join(save_dir, files[i]), img)
    print('Done!!')


def align_video(vid_path, save_dir, desired_size=256):
    if os.path.exists(save_dir):
        if len(os.listdir(save_dir)) > 50:
            return None
    else:
        os.makedirs(save_dir)

    shapes = []
    imgs = []
    frames = load_video(vid_path)
    for I in frames:
        shape = get_landmarks(I)
        if shape is None:
            continue
        imgs.append(I)
        shapes.append(shape[17:])  # 不包括脸部轮廓

    mean_shape = get_position(desired_size)   # 模板脸(平均脸)

    for i, shape in enumerate(shapes):
        # M = transformation_from_points(np.matrix(shape), np.matrix(mean_shape))
        # img = cv2.warpAffine(imgs[i],
        #                      M[:2],  # [R|t]  2 x 3 仿射变换矩阵
        #                      dsize=(desired_size, desired_size),  # output size: (cols, rows)
        #                      borderMode=cv2.BORDER_TRANSPARENT)
        M = cv2.estimateAffinePartial2D(shape, mean_shape)[0]  # 计算仿射变换矩阵
        img = cv2.warpAffine(imgs[i], M, dsize=(desired_size, desired_size), borderMode=cv2.BORDER_REPLICATE)
        # cv2.imwrite(os.path.join(save_dir, f'{i+1}.jpg'), img)
        cx, cy = mean_shape[-20:].mean(0).astype(np.int32)  # 用标准脸的中心作为旋转中心
        # shape = np.dot(shape, M[:, :2].T) + M[:, -1]  # Nx2
        # cx, cy = np.mean(shape[-20:], axis=0).astype(np.int32)
        mouth = img[int(cy - H / 2): int(cy + H / 2), int(cx - W / 2): int(cx + W / 2), ...].copy()
        cv2.imwrite(os.path.join(save_dir, f'{i+1}.jpg'), mouth)


class MyDataset(Dataset):
    def __init__(self, root_dir):
        # D:\LipData\CMLR\video\s5\20090728\section_6_001.69_005.57 + .mp4
        # D:\LipData\CMLR\video_cropped\s5\20090728\section_6_001.69_005.57\xx.jpg
        self.video_paths = glob.glob(os.path.join(root_dir, 's*', '*', '*.mp4'))
        self.mouth_paths = [os.path.splitext(p)[0].replace('video', 'video_cropped') for p in self.video_paths]

    def __getitem__(self, idx):
        align_video(self.video_paths[idx], self.mouth_paths[idx])
        return 0

    def __len__(self):
        return len(self.video_paths)


def run():
    root_dir = r'D:\LipData\CMLR\video'
    loader = DataLoader(MyDataset(root_dir),
                        batch_size=128,
                        num_workers=16,
                        shuffle=False,
                        drop_last=False,
                        pin_memory=True,
                        persistent_workers=True)
    for _ in tqdm(loader):
        pass
    print('Done!!!')


if __name__ == '__main__':
    # align_img('src_faces', 'tgt_faces')
    align_video(r"D:\LipData\CMLR\video\s5\20090728\section_6_001.69_005.57.mp4", 'align_faces')
    # align_video(r"me.mp4", 'align_faces')

    # run()