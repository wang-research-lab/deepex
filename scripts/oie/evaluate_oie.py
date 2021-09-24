import argparse
from deepex.utils import *

def SysCall(command):
    subprocess.Popen(
        command,
        shell=True
    ).wait()

def TopK(Ks, task):
    for k in Ks:
        with open(f"supervised-oie/supervised-oie-benchmark/systems_output/deepex.{task.lower()}.{k}.txt","w") as W:
            c = 0
            with open(f"supervised-oie/supervised-oie-benchmark/systems_output/deepex.{task.lower()}.txt","r") as R:
                for line in R:
                    data = line.strip().split('\t')
                    if len(data) == 1:
                        c = 0; W.write(line)
                    elif len(data) == 5 and c < k:
                        c += 1; W.write(line)

def BuildEvaluationScript(Ks, task):
    config = LoadJSON(f"tasks/configs/{task}.json")
    with open(f"supervised-oie/supervised-oie-benchmark/evaluate.{task.lower()}.sh","w") as W:
        W.write(
"""
mkdir -p ./eval_data/
mkdir -p ./eval_log/
mkdir -p ./eval_data/{0}/
mkdir -p ./eval_log/{0}/
""".format(task)
        )

        for k in Ks:
            W.write(
"""
python3 benchmark.py --gold={0} --out={1} --clausie={2}
echo "{4}"
""".format(config['gold'],
    f"eval_data/{task}/deepex.{task.lower()}.{k}.dat",
    f"systems_output/deepex.{task.lower()}.{k}.txt",
    f"eval_log/{task}/deepex.{task.lower()}.{k}.log",
    f"{task} (top {k})",
)
            )

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', dest='dir', type=str)
    parser.add_argument('-task', dest='task', type=str)
    args = parser.parse_args()
    reserves = set([FILE.split('_')[0] for FILE in os.listdir(args.dir)])
    assert(len(reserves)==1)
    for part in reserves:
        result = LoadJSON(args.dir+f"{part}_result.json")
        data = LoadJSON(args.dir+f"{part}_data.jsonl",jsonl=True)
        with open(f"supervised-oie/supervised-oie-benchmark/systems_output/deepex.{args.task.lower()}.txt","w") as f:
            for ID,sentence in enumerate(data,1):
                strID = '0' * (40 - len(str(ID))) + str(ID)
                f.write(sentence['text']+"\n")
                if strID in result.keys():
                    for triple in result[strID]:
                        f.write(
                            str(ID)+'\t'+
                            ('"'+sentence['text'][triple['subject_char_span'][0]:triple['subject_char_span'][1]]+'"')+'\t'+
                            ('"'+triple['relation']+'"')+'\t'+
                            ('"'+sentence['text'][triple['object_char_span'][0]:triple['object_char_span'][1]]+'"')+'\t'+
                            str(triple['score'] if args.dir.endswith(".unsort/") else -triple['contrastive_dis'])+'\n'
                        )
    K = [3] if args.task=='OIE_2016' else [1]
    TopK(K,task=args.task)
    BuildEvaluationScript(K,task=args.task)
    SysCall(
        f"cp -rf scripts/oie/* supervised-oie/supervised-oie-benchmark/"
    )
    SysCall(
        f"cd supervised-oie/supervised-oie-benchmark/ && bash evaluate.{args.task.lower()}.sh"
    )