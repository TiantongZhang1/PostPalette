# 已暂停：COCO 文本条件场景扩散实验

本目录是从零训练的 COCO 文本条件场景扩散模型（随机初始化 RGB U-Net + 文本条件），
当前已暂停，网页主线不调用这里的任何代码。

暂停原因：
- 在开发机（Windows / Blackwell GPU）上训练和采样时反复出现 CUDA / cuDNN 执行错误和显示驱动重置，
  本地 GPU 工作负载因此暂停。
- 原生输出只有 64×64，初次 1–2 小时训练不能保证遵循复杂描述，早期采样有明显噪声伪影。
- 小组改为由外部 GPT Image API 生成场景配图（见根目录 `image_api.py`），原 `/api/scene` 接口已删除。

当前主线（`server.py`、`models.py`、`scratch/prompt_model.py` 及 emotion / cnn / vae 推理）
不 import 本目录的任何模块。

唯一的间接关系：`coco_prepare.py` 生成的 `data/coco/{train,val,test}.json`
是 GRU 提示词模型训练（`scratch/prompt_train.py`）的输入数据。只做推理时不需要它。
需要重新准备这份数据时运行：

    python -m scratch.experiments.coco_prepare --train-images 20000 --val-images 500 --test-images 500

| 文件 | 作用 |
|---|---|
| `coco_prepare.py` | 下载 COCO 2017 标注和图片，划分 train/val/test |
| `coco_features.py` | 用 Qwen 预计算描述的语言特征 |
| `photo_diffusion.py` | U-Net 结构、噪声计划、DDPM 采样 |
| `coco_train.py` | 扩散模型训练 |
| `run_scene_training.py` | 分段训练启动器（规避上述 cuDNN 错误） |
| `coco_evaluate.py` | 测试集评估 |

原有的训练命令和参数记录见根目录 `README.txt` 末尾的“历史实验”一节。
所有命令的模块路径已从 `scratch.coco_*` 改为 `scratch.experiments.coco_*`。
