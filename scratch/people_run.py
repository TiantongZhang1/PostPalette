"""Continue the authorized people pipeline serially; stop on any failed GPU stage."""
import json,subprocess,sys,time
from pathlib import Path
from .config import ROOT,ARTIFACTS,DATA


def main():
    directory=ARTIFACTS/'people_pipeline';directory.mkdir(exist_ok=True)
    def status(stage):
        (directory/'status.json').write_text(json.dumps({'stage':stage,'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}),encoding='utf-8')
        print(stage,flush=True)
    def run(stage,args):
        if (directory/'STOP').exists():status('paused');raise SystemExit(0)
        status(stage)
        with (directory/(stage+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-X','utf8','-m',*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:status(stage+'_FAILED');raise SystemExit(result.returncode)
    status('waiting_for_expression_training')
    started=time.monotonic()
    while not (ARTIFACTS/'expression/history.json').exists():
        if (directory/'STOP').exists():status('paused');raise SystemExit(0)
        if time.monotonic()-started>25*60:status('expression_wait_timeout');raise SystemExit(1)
        time.sleep(2)
    run('expression_test',['scratch.people_train','expression','--evaluate','--device','cpu'])
    run('action_training',['scratch.people_train','action','--minutes','25','--steps','15000','--batch-size','16'])
    run('action_test',['scratch.people_train','action','--evaluate','--device','cpu'])
    status('waiting_for_people_images')
    started=time.monotonic()
    while not (DATA/'people/manifest.json').exists():
        if (directory/'STOP').exists():status('paused');raise SystemExit(0)
        if time.monotonic()-started>30*60:status('people_data_timeout');raise SystemExit(1)
        time.sleep(2)
    run('person_training',['scratch.person_train','--minutes','60','--steps','30000','--batch-size','8'])
    run('person_test',['scratch.person_train','--evaluate','--device','cpu'])
    status('complete')


if __name__=='__main__':main()
