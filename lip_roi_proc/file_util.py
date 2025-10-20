import os
import shutil
import glob

def clean_dir(root):
    filer = glob.glob(os.path.join(root, 's*', '*'))
    max_len = 0
    N = 0
    for dir in filer:
        files = os.listdir(dir)
        files = [f for f in files if f.endswith('.xy')]
        max_len = max(max_len, len(files))
        #if len(files) > 75:
        #    print(dir, len(files))
        #    N += 1
        if len(files) > 75:
            print(dir)
            shutil.rmtree(dir)
    print('Done')
    print(N)



def check_unseen_test(root):
    unseens = []
    with open('unseen_val.txt', 'r') as f:
        for line in f:
            unseens.append(line.strip())
            #unseens.append(os.path.split(line.strip())[1])

    for spk in ['s1', 's2', 's20', 's22']:
        spk_dir = os.path.join(root, spk)
        uttr_dir = os.listdir(spk_dir)
        for x in uttr_dir:
            xx = os.path.join(spk, x)
            if xx not in unseens:
                print(xx)
                shutil.rmtree(os.path.join(spk_dir, x))


clean_dir('./faces-small/')
#check_unseen_test('faces-small/')




