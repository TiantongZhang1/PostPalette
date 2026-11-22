PostPalette · 网络梗图推荐与涂鸦贴纸生成

当前版本采用“预训练 LLM + 三个从零训练的模型”。
准备代码和数据不等于模型已训练。查看 /api/health 可以确认训练权重是否存在。

环境安装（Windows / Python 3.12 / PowerShell）
1. 在 PostPalette 文件夹打开终端。
2. python -m venv .venv
3. NVIDIA CUDA 环境先安装适配的 PyTorch，例如本项目开发机：
   .\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
4. .\.venv\Scripts\python.exe -m pip install -r requirements.txt

仅准备数据和检查，不训练
   .\.venv\Scripts\python.exe -m scratch.prepare
   .\.venv\Scripts\python.exe -m scratch.check
prepare 不导入 PyTorch，不加载模型。check 只做 CPU 前向与数据检查，不做反向传播或更新权重。
准备脚本会下载 GoEmotions 和六类 QuickDraw，每类先取 8000 个样本。
数据位于 data/，训练产物位于 artifacts/，均不提交 Git。
数据已准备好时直接运行 check；再次 prepare 会按相同规则重建数据。

下面的命令会真正开始训练，请在 GPU 空闲并决定开始时执行
   .\.venv\Scripts\python.exe -m scratch.train cnn --epochs 12
   .\.venv\Scripts\python.exe -m scratch.train vae --epochs 40
   .\.venv\Scripts\python.exe -m scratch.train emotion --epochs 15
建议依次执行。默认设备为 CUDA，CPU 训练可添加 --device cpu。
以上 epoch 数是起始配置，效果和耗时需首次训练后评估。
续训示例：
   .\.venv\Scripts\python.exe -m scratch.train vae --epochs 60 --resume
续训保持原数据、seed、学习率和 batch size。last.pt 保存最后完整 epoch，
best.pt 按验证集选择。中断发生在 epoch 中间时，从上一个完整 epoch 继续。
每次新训练从随机权重初始化，文本词表只从训练集构建，不加载外部词向量。

训练后单独评估（这些命令不更新权重）
   .\.venv\Scripts\python.exe -m scratch.train cnn --evaluate
   .\.venv\Scripts\python.exe -m scratch.train vae --evaluate
   .\.venv\Scripts\python.exe -m scratch.train emotion --evaluate
测试集不参与训练和选模。评估结果写入各任务的 test_metrics.json。
emotion 报告 macro/micro F1 和各类别 F1，CNN 报告准确率和 macro F1，
VAE 报告每张图的 BCE、KL 和归一化总损失。VAE 的评估重建使用潜变量均值，
用于检查重建而非完整的生成质量评估。history.json 保存逐轮训练/验证指标与耗时。
VAE 的 samples.png 显示固定随机种子的先验采样，行顺序对应六个数据类别。
CNN 类别置信度仅衡量类别识别倾向，美观程度需另做人工评价。

2026-11-21 本机首轮训练结果
CNN 训练 12 轮，测试集 4800 张，准确率 0.963958，macro F1 为 0.964001。
情绪 Encoder 训练 15 轮，按验证 macro F1 选中第 8 轮，测试集 5421 条，
macro F1 为 0.548626，micro F1 为 0.598926。后期有过拟合，情绪判断仍会出错。
条件 VAE 训练 40 轮，测试集归一化重建加 KL 损失为 0.263622，
每张图 BCE 为 179.712589，KL 为 26.966696。这些数值不代表准确率。
采样图位于 artifacts/vae/samples.png，仍有模糊、变形和类别混淆。
三个模型均由 scratch/networks.py 随机初始化，训练权重及原始记录保存在 artifacts/。

启动应用
   .\.venv\Scripts\python.exe server.py
打开 http://127.0.0.1:8765/。Ctrl+C 停止服务。
情绪 Encoder、CNN 和 VAE 在 CPU 上推理。Qwen 默认也在 CPU 上推理。
若需让允许的 LLM 使用 GPU，在启动服务前执行：
   $env:POSTPALETTE_LLM_DEVICE = 'cuda'
训练完成或权重更换后重启服务，清除进程内模型缓存。
缺少权重时页面显示“等待模型训练”。取消情绪辅助可使用仅 Qwen 的梗图推荐，
生成涂鸦需要 VAE 和 CNN 两个训练产物。

使用流程
1. 输入帖子或点击“试用示例”，推荐梗图。
2. Qwen 返回三张 Imgflip 梗图、一个涂鸦类别和可编辑/复制的英文配文。
3. 六类涂鸦可手动选择：笑脸、猫、咖啡杯、太阳、花、生日蛋糕。
4. 选择颜色、种子以及配图/透明贴纸，生成并下载 PNG。
图像生成使用 28×28 的模型输出，放大并排版为 512×512。配文单独复制，
不作为生成模型输入，也不会自动写进 PNG。颜色和透明背景通过普通图像处理实现。

课程主题与模块
- Transformer Encoder：随机初始化的两层 Encoder + 训练集词表及随机词向量，
  在 GoEmotions 合并后的八类情绪上训练，采用多标签 BCE 损失。
- LLM：Qwen/Qwen2.5-1.5B-Instruct，是唯一加载外部预训练权重的模型。
  处理中文翻译、候选梗图排序、涂鸦类别选择和短配文。
- Traditional GenAI：随机初始化条件 VAE，在 QuickDraw 六类图像上训练，
  用 BCE 重建损失与 KL 正则学习可按类别采样的潜空间。
- CNN：独立随机初始化的卷积分类器，识别六类涂鸦。VAE 每次生成 16 个候选，
  CNN 按目标类别分数排序选出一个。与 VAE 共用固定的数据划分。

数据来源与处理记录
GoEmotions（Google Research，Apache-2.0）
https://github.com/google-research/google-research/tree/master/goemotions
保留官方 train/dev/test 边界，删除完全相同的标准化文本，测试/验证优先保留。
将 28 个原标签按 scratch/config.py 合并成八组，保留多标签。
这种标签合并是项目设计选择，八组任务指标不能直接和原始 28 类榜单比较。
情绪输出是模型分数，未经置信度校准。

Quick, Draw!（Google Creative Lab，CC BY 4.0）
https://github.com/googlecreativelab/quickdraw-dataset
https://creativecommons.org/licenses/by/4.0/
从官方 28×28 位图流式读取每类前 8000 张，全局去除完全相同的位图，
每类打乱后按 80/10/10 划分 train/val/test。这是前缀抽样，未覆盖完整数据分布。
类别顺序为 smiley face、cat、coffee cup、sun、flower、birthday cake。
每份 manifest.json 保存来源 URL、下载内容哈希、样本数、时间和划分规则。
训练权重记录数据指纹、随机初始化标记、训练步数、配置和验证历史。
这两个数据集没有帖子—梗图相关性标签，梗图相关性需要另建人工评估样例。

梗图来源
https://imgflip.com/api
从公开热门模板目录读取图片地址和热度，用途标注和关键词筛选后由 Qwen 排序。
目录请求失败时用内置模板元数据，缩略图仍需联网。不向 Imgflip 发送帖子内容。
图片显示来源链接，数据集许可不等同于第三方梗图图片许可。

Chrome 扩展
保持服务运行在默认端口 8765。在 chrome://extensions/ 开启开发者模式，
加载 extension/ 文件夹。在普通网页选中文字，再点 PostPalette 图标。
浏览器设置页等特殊页面可直接在工作台粘贴文字。

分享项目
分享源码时排除 .venv、.model-cache、data、artifacts 和 __pycache__。
团队复现需要运行数据准备与训练，或单独共享本组训练产物及其 manifest/history。
旧版 .model-cache 中的其他预训练权重不会被当前代码使用。
