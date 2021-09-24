import os
import shutil
import argparse
import subprocess
from deepex.utils import *
from requests import get

def SysCall(command):
    subprocess.Popen(
        command,
        shell=True
    ).wait()

def PreprocessData(META_TASK, RAW_PATH, DATA_PATH):
    if META_TASK in ["OIE_2016"]:
        Create(DATA_PATH)
        for i, t in enumerate(['test', 'dev']):
            data = []; file_path = RAW_PATH+f"{t}.txt"
            if os.path.exists(file_path):
                with open(RAW_PATH+f"{t}.txt","r") as f:
                    for j,line in enumerate(f):
                        data.append(
                            {
                                "id": str(j+1),
                                "title": str(j+1),
                                "text": line[:-1].replace('(',' ').replace(')',' '),
                            }
                        )
            elif t=='test':
                raise Exception(f"Test data files not found at '{file_path}'!")
            SaveJSON(data,DATA_PATH+f"P{i}.jsonl",jsonl=True)
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

if __name__=="__main__":
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument("-t", "--task", dest="task", type=str, default='OIE_2016',
        choices=[
            'OIE_2016',
            'WEB',
            'NYT',
            'PENN',
            'FewRel',
            'TACRED',
        ],
        help = "The task to be run"
    )
    parser.add_argument("-m", "--model", dest="model", type=str, default='bert-large-cased',
        choices=[
            'bert-base-cased',
            'bert-large-cased',
        ],
        help = "The pre-trained model type to be used for generating attention matrices to perform beam search on"
    )
    parser.add_argument("-q", "--beam-size", dest="beam_size", type=int, default=6,
        help = "The beam size during beam search"
    )
    parser.add_argument("-k", "--max-distance", dest="max_distance", type=int, default=2048,
        help = "The maximum distance allowed between entities during beam search"
    )
    parser.add_argument("-b", "--batch-size-per-device", dest="batch_size_per_device", type=int, default=4,
        help = "The batch size on a single device",
    )
    parser.add_argument("-s", "--stage", dest="stage", type=int, default=0,
        help = "Run task starting from an intermediate stage:\n0). data preparation and beam-search\n1). post processing\n2). ranking\n3). evaluation",
    )
    parser.add_argument("-d", "--debug", dest="debug", action="store_const", const=True, default=False,
        help="If true, only the specified stage will be run, otherwise successive stages will be run."
    )
    parser.add_argument("-c", "--clean-history", dest="clean_history", action="store_const", const=True, default=False,
        help="If true, clean history runs."
    )
    parser.add_argument("-p", "--prepare-rc-dataset", action="store_const", const=True, default=False,
        help="If true, automatically run the rc dataset preparation scripts."
    )
    parser.add_argument("--cuda", dest="cuda", type=str, default="0,1,2,3,4,5,6,7",
        help="Specify CUDA gpu devices."
    )
    args = parser.parse_args(); CUR_DIR = os.getcwd()+"/"
    config = LoadJSON("tasks/configs/"+args.task+".json")
    args.batch_size = args.batch_size_per_device*len(args.cuda.split(','))
    args.task_abbr = config['task_abbr']
    args.task_meta = config['task_meta']
    args.data_dir  = config['data_dir' ]
    args.proc_dir  = "output/data/"+args.task_meta+"/"
    args.outp_dir  = "output/output/"+args.task_meta+"/"
    args.clss_dir  = "output/classified/"+args.task_meta+"/"
    args.beam_mode = 'IE' if config['task_abbr']=='oie' else 'RC'
    args.ner_mode  = 'np' if config['task_abbr']=='oie' else 'rc'
    args.part      = '0'
    
    if args.clean_history and args.stage==0 and os.path.exists("output/"):
        shutil.rmtree("output/"); shutil.rmtree("runs/")
    Create("runs/")
    Create("result/")
    Create("output/")
    Create("output/data/")
    Create("output/output/")
    Create("output/classified/")
    Create(args.proc_dir)
    Create(args.outp_dir)
    Create(args.clss_dir)

    if args.stage<=0 and (args.stage==0 or not args.debug):
        if args.prepare_rc_dataset:
            assert (args.task in ['FewRel','TACRED']), ("Only task 'FewRel' and 'TACRED' support `--prepare-rc-dataset` argument!")
            if args.task=='TACRED':
                assert (os.path.exists("./tacred_LDC2018T24.tgz")), ("Please first download TACRED datase according to README.md! The downloaded file should be named as `tacred_LDC2018T24.tgz`.")
            SysCall(
                "bash scripts/rc/prep_{}.sh".format(args.task)
            )
        PreprocessData(args.task_meta,args.data_dir,args.proc_dir)
        if args.part is not None:
            reserves = set(["P"+i+".jsonl" for i in args.part.split(',')])
            for FILE in os.listdir(args.proc_dir):
                if FILE not in reserves:
                    os.remove(args.proc_dir+FILE)
        SysCall(
            ("bash scripts/processing.sh %s 1266 {1} None {2} {3} {4} {8} fast_unsupervised_bidirectional_beam_search 256 score_len 1 mean sum 1 {7} {6} {5}"%args.cuda)
            .format(args.task_abbr,args.proc_dir,args.outp_dir,args.model.split('-')[0]+" "+args.model,args.ner_mode,args.beam_mode,args.beam_size,args.max_distance,args.batch_size)
        )

    if args.stage<=1 and (args.stage==1 or not args.debug):
        for FOLDER in os.listdir(args.outp_dir):
            model, _, ner_mode, _, _, _, _, _, _, beam_size = FOLDER.split('.')
            if (model==args.model) and (ner_mode==args.ner_mode) and (int(beam_size)==args.beam_size):
                reserves = set([str(i) for i in args.part.split(',')]); classes = set()
                for BATCH in os.listdir(args.outp_dir+FOLDER+"/"):
                    if BATCH != "run.log":
                        part = BATCH.split('_')[0]; batch_folder = args.outp_dir+FOLDER+"/"+BATCH+"/"
                        if part in reserves:
                            classified = args.clss_dir+"P"+part+"/"; Create(classified)
                            SysCall(
                                "cp -r {0} {1}"
                                .format(batch_folder,classified)
                            )
                            classes.add(classified)
                for classified in classes:
                    SysCall(
                        ("bash scripts/post_processing.sh %s 1266 {1} None {2} {3} {4} {8} fast_unsupervised_bidirectional_beam_search 256 score_len 1 mean sum 1 {7} {6} {5}"%args.cuda)
                        .format(args.task_abbr,args.proc_dir,classified,args.model.split('-')[0]+" "+args.model,args.ner_mode,args.beam_mode,args.beam_size,args.max_distance,args.batch_size)
                    )

    if args.stage<=2 and (args.stage==2 or not args.debug):
        RESULT = "result/" + ".".join([args.task,args.model,args.ner_mode,f"d{args.max_distance}",f"b{args.beam_size}"])
        SysCall(
            "python3 scripts/ranking.py -proc_dir {0} -clss_dir {1} -dest {2}".format(args.proc_dir,args.clss_dir,RESULT+".unsort")
        )
        SysCall(
            "python3 scripts/ranking.py -proc_dir {0} -clss_dir {1} -dest {2}".format(args.proc_dir,args.clss_dir,RESULT+".sorted")
        )

    if args.stage<=3 and (args.stage==3 or not args.debug) and (args.task_meta in ['OIE_2016','WEB','NYT','PENN']):
        RESULT = "result/" + ".".join([args.task,args.model,args.ner_mode,f"d{args.max_distance}",f"b{args.beam_size}"])
        SysCall(
            "python3 scripts/oie/evaluate_oie.py -dir {0} -task {1}".format(RESULT+".sorted/",args.task)
        )

    if args.stage<=3 and (args.stage==3 or not args.debug) and (args.task_meta in ['FewRel', 'TACRED']):
        RESULT = "result/" + ".".join([args.task,args.model,args.ner_mode,f"d{args.max_distance}",f"b{args.beam_size}"]); Create("scripts/rc/data/")
        SysCall(
            "cp {0}P0_data.jsonl scripts/rc/data/{1}_data.jsonl".format(RESULT+".sorted/",args.task_meta.lower())
        +   "&& cp {0}P0_result.json scripts/rc/data/{1}_result.json".format(RESULT+".sorted/",args.task_meta.lower())
        +   "&& bash scripts/rc/eval_" + args.task_meta + ".sh"
        )
