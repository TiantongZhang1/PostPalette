"""CPU analysis of person candidates using only this project's trained models."""
from functools import cache
from io import BytesIO
import numpy as np
from PIL import Image,ImageOps
import torch
from models import ModelError,inference_lock
from scratch.config import ARTIFACTS
from scratch.photo_tags import decode_photo
from scratch.people_networks import PersonPoseNet,ImageClassifier,detections

ACTION_NAMES={'applauding':'鼓掌','blowing_bubbles':'吹泡泡','brushing_teeth':'刷牙','cleaning_the_floor':'清洁地面',
'climbing':'攀爬','cooking':'做饭','cutting_trees':'砍树','cutting_vegetables':'切菜','drinking':'喝东西','feeding_a_horse':'喂马',
'fishing':'钓鱼','fixing_a_bike':'修自行车','fixing_a_car':'修车','gardening':'园艺','holding_an_umbrella':'撑伞',
'jumping':'跳跃','looking_through_a_microscope':'看显微镜','looking_through_a_telescope':'看望远镜','playing_guitar':'弹吉他',
'playing_violin':'拉小提琴','pouring_liquid':'倒液体','pushing_a_cart':'推车','reading':'阅读','phoning':'打电话',
'riding_a_bike':'骑自行车','riding_a_horse':'骑马','rowing_a_boat':'划船','running':'跑步','shooting_an_arrow':'射箭',
'smoking':'吸烟','taking_photos':'拍照','texting_message':'发消息','throwing_frisby':'扔飞盘','using_a_computer':'使用电脑',
'walking_the_dog':'遛狗','washing_dishes':'洗碗','watching_TV':'看电视','waving_hands':'挥手',
'writing_on_a_board':'在板上写字','writing_on_a_book':'书写'}
EXPRESSION_NAMES={'happy':'笑容倾向','neutral':'中性表情倾向','surprise':'惊讶样式表情','angry':'愤怒样式表情',
                  'disgust':'厌恶样式表情','fear':'恐惧样式表情','sad':'悲伤样式表情'}


def model_status():
    return {key:(ARTIFACTS/key/'best.pt').is_file() and (ARTIFACTS/key/'test_metrics.json').is_file()
            for key in ('person_pose','action','expression')}


@cache
def load_model(task):
    path=ARTIFACTS/task/'best.pt'
    if not path.exists():raise ModelError('人物分析模型正在准备或训练，暂时不能自动分析。')
    record=torch.load(path,map_location='cpu',weights_only=True)
    if record.get('task')!=task or record.get('initialization')!='random' or record.get('step',0)<1:
        raise ModelError('人物分析检查点缺少有效的自训记录。')
    model=PersonPoseNet() if task=='person_pose' else ImageClassifier(**record['config'])
    model.load_state_dict(record['model'],strict=True)
    return model.cpu().eval(),record


def tensor(image,size,gray=False):
    image=image.convert('L' if gray else 'RGB')
    image=ImageOps.pad(image,(size,size),Image.Resampling.LANCZOS,color=127 if gray else (127,127,127))
    pixels=np.array(image)
    pixels=pixels[None] if gray else pixels.transpose(2,0,1)
    return torch.from_numpy(pixels.copy())[None].float()/255


@torch.inference_mode()
def classify(task,image):
    model,record=load_model(task)
    probabilities=model(tensor(image,48 if task=='expression' else 128,task=='expression')).softmax(1)[0]
    values,indices=probabilities.topk(min(3,len(probabilities)))
    names=EXPRESSION_NAMES if task=='expression' else ACTION_NAMES
    results=[{'label':record['labels'][i],'name':names.get(record['labels'][i],record['labels'][i]),'score':float(v)} for v,i in zip(values,indices)]
    reliable=results[0]['score']>=.7 and results[0]['score']-results[1]['score']>=.2
    return {'candidates':results,'uncertain':not reliable,'step':record['step']}


def pose_hint(points):
    if all(points[i][2]>.65 for i in (5,6,9,10)):
        if points[9][1]<points[5][1] or points[10][1]<points[6][1]:return '可能有举手姿态'
    if all(points[i][2]>.65 for i in (11,12,13,14,15,16)):
        hips=np.mean([points[i][:2] for i in (11,12)],0);knees=np.mean([points[i][:2] for i in (13,14)],0)
        ankles=np.mean([points[i][:2] for i in (15,16)],0)
        if abs(knees[1]-hips[1])<.65*abs(knees[0]-hips[0]) and ankles[1]>knees[1]:return '可能为坐姿或屈膝姿态'
        if knees[1]>hips[1] and ankles[1]>knees[1] and abs(knees[0]-hips[0])<.4*(ankles[1]-hips[1]):return '可能为直立姿态'
    return None


def analyze_people(content):
    decode_photo(content)  # Shared file-format, size and decompression validation.
    if not model_status()['person_pose']:
        raise ModelError('人物检测的训练或独立评估尚未完成，请稍后再试。')
    with Image.open(BytesIO(content)) as original:
        rgba=ImageOps.exif_transpose(original).convert('RGBA')
        image=Image.alpha_composite(Image.new('RGBA',rgba.size,(127,127,127,255)),rgba).convert('RGB')
    with inference_lock,torch.inference_mode():
        torch.set_num_threads(4);model,record=load_model('person_pose');size=record['size']
        if record['best_val']<.35:raise ModelError('人物检测模型的验证效果还不足，暂不输出人数；请继续训练或手动补充。')
        people=detections(model(tensor(image,size)),size,record['threshold'])[0]
        scale=size/max(image.size);px=(size-round(image.width*scale))//2;py=(size-round(image.height*scale))//2
        result=[];truncated=False
        for person in people:
            box=person['box'];coordinates=[(box[0]-px)/scale,(box[1]-py)/scale,(box[2]-px)/scale,(box[3]-py)/scale]
            x1,y1,x2,y2=coordinates;x1=max(0,x1);y1=max(0,y1);x2=min(image.width,x2);y2=min(image.height,y2)
            if x2-x1<5 or y2-y1<5:continue
            if len(result)>=10:
                truncated=True;break
            w=x2-x1;h=y2-y1
            # Stanford40 remains an archived experiment. Whole-photo activities use their own endpoint.
            action=None
            pose_supported=record['history'][-1]['val']['matched_person_pck_02']>=.5
            expression=None;face_region=None;points=person['keypoints']
            # Do not classify the entire body as a face. Require eye evidence and a sufficiently large crop.
            if pose_supported and all(points[i][2]>.8 for i in (0,1,2)) and model_status()['expression']:
                left=np.array(points[1][:2]);right=np.array(points[2][:2]);eye_distance=np.linalg.norm(left-right)/scale
                eyes_in_image=all(px<=point[0]<px+image.width*scale and py<=point[1]<py+image.height*scale for point in (left,right))
                if eyes_in_image and eye_distance>=10 and abs(left[1]-right[1])/scale < w*.2 and max(left[1],right[1])<box[1]+.4*(box[3]-box[1]):
                    center=(left+right)/2;cx=(center[0]-px)/scale;cy=(center[1]-py)/scale
                    fw=eye_distance*2.8;fh=fw*1.25
                    face_box=(max(0,int(cx-fw/2)),max(0,int(cy-fh*.35)),min(image.width,int(cx+fw/2)),min(image.height,int(cy+fh*.65)))
                    if min(face_box[2]-face_box[0],face_box[3]-face_box[1])>=24:
                        face=image.crop(face_box)
                        expression=classify('expression',face)
                        face_region=[face_box[0]/image.width,face_box[1]/image.height,face_box[2]/image.width,face_box[3]/image.height]
            result.append({'number':len(result)+1,'box':[x1/image.width,y1/image.height,x2/image.width,y2/image.height],
                           'score':person['score'],'pose_hint':pose_hint(points) if pose_supported else None,'action':action,'expression':expression,'expression_region':face_region})
        count=len(result)
        description=f'检测到{"至少 " if truncated else ""}{count} 个人物候选（可能漏检或重复）。'
        if count==2:description+='两个人同框，是否为合影需要确认。'
        return {'people':result,'count_estimate':count,'truncated':truncated,'summary':description,'step':record['step'],'device':'cpu',
                'notice':'人数、姿态和表情均为候选结果。照片无法证明人物关系或真实内心情绪；看不清人脸时不分析表情。'}
