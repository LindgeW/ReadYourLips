import cv2
import time
import os
from threading import Thread


def load_frames(path):
    def read_img(idx, f):
        imgs.append((idx, cv2.imread(f)))

    t1 = time.time()
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    imgs = [] 
    #imgs = [cv2.imread(os.path.join(path, f)) for f in files]
    #print(imgs[0].shape, len(imgs))
    ths = []
    for i, f in enumerate(files):
        th = Thread(target=read_img, args=(i, f, ))
        th.start()
        ths.append(th)
    for th in ths:
        th.join()
    
    imgs = sorted(imgs, key=lambda x: x[0])
    imgs = list(map(lambda x: x[0], imgs))
    print(imgs)
    t2 = time.time()
    print(f'time cost: {t2 - t1}s')


load_frames('faces/s1/swwv9a')
load_frames('faces/s22/swwt8a')
