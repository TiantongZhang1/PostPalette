"""Small, randomly initialized whole-photo activity CNN and shared preprocessing."""
from io import BytesIO
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from .people_networks import Residual

LABELS = ('eating', 'drinking', 'running', 'cycling', 'clapping', 'sitting', 'using_laptop', 'hugging')
NAMES = dict(zip(LABELS, ('吃东西', '喝东西', '跑步', '骑自行车', '鼓掌', '坐着', '使用笔记本电脑', '拥抱')))
CLASSES = LABELS + ('other',)
SOURCE_LABELS = ('calling', 'clapping', 'cycling', 'dancing', 'drinking', 'eating', 'fighting',
                 'hugging', 'laughing', 'listening_to_music', 'running', 'sitting', 'sleeping', 'texting', 'using_laptop')
SIZE = 160


class ActivityCNN(nn.Module):
    def __init__(self, classes=9, normalization='group'):
        super().__init__()
        self.features = nn.Sequential(Residual(3,24,2), Residual(24,48,2), Residual(48,96,2),
                                      Residual(96,96), Residual(96,160,2), Residual(160,160))
        self.head = nn.Sequential(nn.Linear(320,160), nn.SiLU(), nn.Dropout(.3), nn.Linear(160,classes))
        if normalization not in ('group','batch'):raise ValueError('Unknown normalization')
        if normalization=='batch':
            def replace(parent):
                for name,child in parent.named_children():
                    if isinstance(child,nn.GroupNorm):setattr(parent,name,nn.BatchNorm2d(child.num_channels))
                    else:replace(child)
            replace(self.features)

    def forward(self,x):
        x = self.features(x)
        return self.head(torch.cat((x.mean((2,3)),x.amax((2,3))),1))


def preprocess(image):
    image = ImageOps.exif_transpose(image).convert('RGBA')
    image = Image.alpha_composite(Image.new('RGBA', image.size, (127,127,127,255)), image).convert('RGB')
    image = ImageOps.pad(image, (SIZE,SIZE), Image.Resampling.LANCZOS, color=(127,127,127))
    return np.array(image).transpose(2,0,1).copy()


def decode(content):
    from .photo_tags import decode_photo
    decode_photo(content)  # Share upload limits, format and decompression validation.
    with Image.open(BytesIO(content)) as image:
        return preprocess(image)


def decide(probabilities, thresholds, margin=.15):
    """Abstain unless a supported class passes its validation-only selection rule."""
    order = np.argsort(-np.asarray(probabilities))
    first, second = int(order[0]), int(order[1])
    accepted = (first < len(LABELS) and float(probabilities[first]) >= thresholds[first]
                and float(probabilities[first])-float(probabilities[second]) >= margin)
    return {'accepted': bool(accepted), 'label': CLASSES[first] if accepted else None,
            'top_index': first, 'order': order[:3].tolist()}


def adjust_probabilities(probabilities, other_bias=0.):
    """One validation-selected logit offset compensates for the sampled 'other' prior."""
    adjusted=np.asarray(probabilities).copy()
    adjusted[..., -1]*=np.exp(other_bias)
    return adjusted/adjusted.sum(axis=-1,keepdims=True)
