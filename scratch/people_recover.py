"""CPU-only continuation after the Windows CUDA execution failure; no GPU retries."""
import json,subprocess,sys,time
from .config import ROOT,ARTIFACTS


def main():
    directory=ARTIFACTS/'people_pipeline'
    def status(stage):
        content={'stage':stage,'device':'cpu','reason':'CUDA action training execution failure; GPU retries disabled',
                 'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
        (directory/'status.json').write_text(json.dumps(content),encoding='utf-8');print(stage,flush=True)
    def run(stage,arguments):
        if (directory/'STOP').exists():status('paused');raise SystemExit(0)
        status(stage)
        with (directory/(stage+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-X','utf8','-m',*arguments],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:status(stage+'_FAILED');raise SystemExit(result.returncode)
    run('action_cpu_resume',['scratch.people_train','action','--resume','--device','cpu','--minutes','10','--steps','12000','--batch-size','16'])
    run('action_test',['scratch.people_train','action','--evaluate','--device','cpu'])
    person_arguments=['scratch.person_train','--device','cpu','--minutes','50','--steps','20000','--batch-size','8']
    if (ARTIFACTS/'person_pose/last.pt').exists():person_arguments.append('--resume')
    run('person_cpu_training',person_arguments)
    run('person_test',['scratch.person_train','--evaluate','--device','cpu'])
    status('complete')


if __name__=='__main__':main()
