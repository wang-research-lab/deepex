from utils import *

def PreprocessData(META_TASK, RAW_PATH, DATA_PATH):
    if META_TASK in ["OIE_2016"]:
        Create(DATA_PATH)
        data = []
        if os.path.exists(RAW_PATH+"test.txt"):
            with open(RAW_PATH+"test.txt","r") as f:
                for j,line in enumerate(f):
                    data.append(
                        {
                            "id": str(j+1),
                            "title": str(j+1),
                            "text": line[:-1].replace('(',' ').replace(')',' '),
                        }
                    )
        SaveJSON(data,DATA_PATH+f"P0.jsonl",jsonl=True)
    elif META_TASK in ["FewRel","TACRED"]:
        Create(DATA_PATH)
        data = LoadJSON(f"data/{META_TASK}/data.jsonl",jsonl=True)
        SaveJSON(data,DATA_PATH+"P0.jsonl",jsonl=True)
    else:
        Create(DATA_PATH); data = []
        with open(RAW_PATH+"{0}.raw".format(META_TASK.lower()),"r") as f:
            for j,line in enumerate(f):
                data.append(
                    {
                        "id": str(j+1),
                        "title": str(j+1),
                        "text": line[:-1].replace('(',' ').replace(')',' '),
                    }
                )
        SaveJSON(data,DATA_PATH+"P0.jsonl",jsonl=True)
