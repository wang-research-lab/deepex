import os
import re
import sys
import math
import time
import tqdm
import json
import shutil
import logging
import requests
import jsonlines
import subprocess
import numpy as np
from tqdm import tqdm

def Folder(PATH):
    return "/".join(PATH.split('/')[:-1])+"/"
    
def File(PATH):
    return PATH.split('/')[-1]

def Prefix(PATH):
    return ".".join(PATH.split('.')[:-1])

def Suffix(PATH):
    return PATH.split('.')[-1]

def Create(PATH):
    None if os.path.exists(PATH) else os.makedirs(PATH)

def Delete(PATH):
    shutil.rmtree(PATH) if os.path.exists(PATH) else None

def Clear(PATH):
    shutil.rmtree(PATH) if os.path.exists(PATH) else None; os.makedirs(PATH)


def SaveJSON(object, FILE, jsonl=False, indent=None):
    if jsonl:
        with jsonlines.open(FILE, 'w') as f:
            for data in object:
                f.write(data)
    else:
        with open(FILE, 'w') as f:
            json.dump(object, f, indent=indent)

def PrettifyJSON(PATH):
    if PATH[-1]=='/':
        for FILE in os.listdir(PATH):
            SaveJSON(LoadJSON(PATH+FILE),PATH+FILE,indent=4)
    else:
        SaveJSON(LoadJSON(PATH),PATH,indent=4)

def LoadJSON(FILE, jsonl=False):
    if jsonl:
        with open(FILE, 'r') as f:
            return [data for data in jsonlines.Reader(f)]
    else:
        with open(FILE, 'r') as f:
            return json.load(f)

def View(something, length=4096):
    print(str(something)[:length]+" ..." if len(str(something))>length+3 else str(something))

def ViewS(something, length=4096):
    return (str(something)[:length]+" ..." if len(str(something))>length+3 else str(something))

def ViewDict(something, length=4096, limit=512):
    print("{")
    for i,item in enumerate(something.items()):
        print("\t"+str(item[0])+": "+(ViewS(item[1])+','))
        if i>=limit:
            print("\t..."); break
    print("}")

def ViewDictS(something, length=4096, limit=512):
    s = "{\n"
    for i,item in enumerate(something.items()):
        s += "\t"+str(item[0])+": "+(ViewS(item[1])+',')+"\n"
        if i>=limit:
            s += "\t...\n"; break
    s += "}\n"; return s

def ViewJSON(json_dict, length=4096):
    print(ViewS(json.dumps(json_dict,indent=4)))

def ViewJSONS(json_dict, length=4096):
    return ViewS(json.dumps(json_dict,indent=4))

def DATE():
    return time.strftime("%Y-%m-%d",time.localtime(time.time()))

def CMD(command, wait=True):
    h = subprocess.Popen(command,shell=True); return h.wait() if wait else h

def PrintConsole(*args, **kwargs):
    print(*args, file=sys.stdout, **kwargs)

def PrintError(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def LineToFloats(line):
    return [float(s) for s in re.findall(r"(?<!\w)[-+]?\d*\.?\d+(?!\d)",line)]

def ERROR(something):
    return "\033[31m"+str(something)+"\033[0m"

def SUCCESS(something):
    return "\033[32m"+str(something)+"\033[0m"

def WARN(something):
    return "\033[33m"+str(something)+"\033[0m"

import wget
def cached_path(url):
    CACHE_DIR = ".cache/"; path = CACHE_DIR+"/".join(url.split('/')[3:])
    os.makedirs("/".join(path.split("/")[:-1])); wget.download(url, out=path); return path