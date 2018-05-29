import bcolz
import os


def save_bcolz(feature, root, name):
    if not os.path.exists(root):
        os.mkdir(root)
    bcolz.carray(feature, chunklen=1024, mode='w', rootdir=f"{root}/{name}")

def load_bcolz(root, name):
    if not os.path.exists(root):
        print(f"[+] Feature {name} does not exists")
        return None
    return bcolz.open(f"{root}/{name}")