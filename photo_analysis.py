"""One upload and one local analysis request, with independent module failures."""
import logging
from models import ModelError
from activity_posts import recognize_activity
from photo_posts import recognize_photo
from people_analysis import analyze_people
from scratch.photo_tags import decode_photo


def analyze_photo(content):
    decode_photo(content)  # Reject invalid uploads before running any model.
    result = {'activity': None, 'objects': None, 'people': None, 'errors': []}
    for key, name, handler in (
        ('activity', '活动', recognize_activity),
        ('objects', '物体', recognize_photo),
        ('people', '人物', analyze_people),
    ):
        try:
            # Each model owns its inference lock; no enclosing lock or GPU work.
            result[key] = handler(content)
        except ModelError as error:
            result['errors'].append({'module': key, 'name': name, 'message': str(error)})
        except Exception:
            logging.exception('Photo analysis module failed: %s', key)
            result['errors'].append({'module': key, 'name': name, 'message': f'{name}分析暂不可用，可以手动补充线索。'})
    result['partial'] = bool(result['errors'])
    result['notice'] = '线索仅在本机分析。请删除误识别、补充遗漏，统一核对后再生成推文。'
    return result
