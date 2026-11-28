"""Local, whole-photo activity suggestions; only user-confirmed text reaches Qwen."""
from functools import lru_cache
import json
import math
import torch
from models import ModelError, inference_lock
from scratch.config import ARTIFACTS
from scratch.daily_activity import ActivityCNN, CLASSES, LABELS, NAMES, SIZE, decode, decide, adjust_probabilities

DIRECTORY = ARTIFACTS/'daily_activity'


def release():
    try:
        checkpoint=DIRECTORY/'best.pt';evaluation=DIRECTORY/'test_metrics.json';calibration=DIRECTORY/'calibration.json'
        report=json.loads(evaluation.read_text());rule=json.loads(calibration.read_text())
        if evaluation.stat().st_mtime_ns<checkpoint.stat().st_mtime_ns:return None
        if report['step']!=rule['step'] or report['step']<1 or report['labels']!=list(CLASSES):return None
        if {key:value for key,value in rule.items() if key!='step'}!=report['calibration']:return None
        if not all(math.isfinite(report[key]) and 0<=report[key]<=1 for key in ('accuracy','macro_f1')):return None
        if len(rule['thresholds'])!=len(LABELS) or not all(math.isfinite(t) and 0<=t<=1.01 for t in rule['thresholds']):return None
        if not 0<=rule['margin']<=1:return None
        if not math.isfinite(rule.get('other_logit_bias',0)) or abs(rule.get('other_logit_bias',0))>3:return None
        return report,rule,(checkpoint.stat().st_mtime_ns,evaluation.stat().st_mtime_ns,calibration.stat().st_mtime_ns)
    except (OSError,ValueError,KeyError,TypeError):return None


def activity_status():
    bundle=release()
    automatic=[key for i,key in enumerate(LABELS) if bundle and bundle[1]['thresholds'][i]<=1]
    return {'ready':bundle is not None,'labels':[{'id':key,'name':NAMES[key],'auto_supported':key in automatic} for key in LABELS],
            'manual_only':[{'id':key,'name':NAMES[key]} for key in LABELS if key not in automatic],
            'device':'cpu','scope':'whole_photo_dominant_activity','size':SIZE,
            'test_accuracy':bundle[0]['accuracy'] if bundle else None,
            'test_macro_f1':bundle[0]['macro_f1'] if bundle else None}


@lru_cache(maxsize=1)
def load_model(signature):
    record=torch.load(DIRECTORY/'best.pt',map_location='cpu',weights_only=True)
    bundle=release()
    if (bundle is None or bundle[2]!=signature or record.get('task')!='daily_activity'
            or record.get('initialization')!='random' or record.get('size')!=SIZE
            or list(record.get('labels',()))!=list(CLASSES) or record.get('step')!=bundle[0]['step']
            or record['manifest']['parquet_sha256']!=bundle[0]['parquet_sha256']):
        raise ModelError('日常活动模型的训练、校准与测试记录不一致，请完成评估后再试。')
    model=ActivityCNN(**record.get('config',{'normalization':'group'})).cpu().eval();model.load_state_dict(record['model'],strict=True)
    return model,record


def recognize_activity(content):
    pixels=decode(content)
    bundle=release()
    if bundle is None:raise ModelError('日常活动模型正在训练或评估，暂时可以手动选择活动后写推文。')
    report,rule,signature=bundle
    with inference_lock,torch.inference_mode():
        torch.set_num_threads(4);model,record=load_model(signature)
        probabilities=model(torch.from_numpy(pixels)[None].float()/255).softmax(1)[0].numpy()
        probabilities=adjust_probabilities(probabilities,rule.get('other_logit_bias',0))
        result=decide(probabilities,rule['thresholds'],rule['margin'])
    candidates=[{'id':CLASSES[i],'name':NAMES.get(CLASSES[i],'其他活动／不属于支持范围'),'score':float(probabilities[i])} for i in result['order']]
    return {'suggestion':result['label'],'accepted':result['accepted'],'candidates':candidates,'step':record['step'],'device':'cpu',
            'scope':'whole_photo_dominant_activity',
            'notice':'仅判断整张照片的主要活动，不判断每个人的动作、人数、人物关系或真实心情。分数不是正确概率，请核对后使用。',
            'uncertain_reason':None if result['accepted'] else '无法可靠判断，或照片不属于这 8 类活动；可以手动选择或补充背景。'}
