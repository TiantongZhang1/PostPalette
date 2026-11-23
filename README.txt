PostPalette · 网络梗图推荐、自训提示词与 GPT 场景配图

当前默认方案（2026-11-22）
用户帖子 → 自训 GRU 注意力/复制编码器解码器 → 可编辑提示词 → OpenAI GPT Image API。
本地所有推理和新增提示词训练固定使用 CPU，不启动 GPU。原 COCO 扩散训练保持暂停。
课程边界：按小组当前选择接入 GPT 生图；课件列出 ChatGPT 等 LLM，但未明确图像能力的例外。
因此本 README 不宣称教授已批准外部生图，也不把 API 输出计作从零训练的图像模型成果。

快速使用
1. 已有虚拟环境则跳过安装，首次安装按下文环境安装说明操作。
2. 复制配置：Copy-Item .env.example .env
   在本机编辑 .env，填写 OPENAI_API_KEY。不要发到聊天、截图、Git 或组员压缩包里。
   默认 POSTPALETTE_IMAGE_MODEL=gpt-image-1.5，质量 medium，可改 low/medium/high/auto。
   环境变量优先于 .env；修改后重启服务。API 请求按用量计费，ChatGPT 订阅与 API 额度分开。
3. .\.venv\Scripts\python.exe server.py
   打开 http://127.0.0.1:8765/，输入帖子，点击“根据帖子生成提示词”。
4. 检查模型草稿和可编辑提示词，点击“生成图片”，再下载 PNG。
   提示词生成不调用付费 API；只有点击生图才发送当前提示词至 OpenAI。
   中文先由本地 Qwen 翻译，可能较慢；英文直接进入自训模型。
   页面允许手动输入提示词，用于与自训提示词做对照；这不计作自训模型输出。
   GPT Image 不提供本项目可用的种子控制，种子仅适用于原涂鸦模式。

提示词模型的真实训练内容
scratch/prompt_model.py：随机词向量 + GRU Encoder/Decoder + 点积注意力 + 复制概率混合。
没有加载预训练文本编码器、词向量或其他非 LLM 权重。
scratch/prompt_train.py：CPU 上用 COCO 2017 人工描述做去噪序列生成，随机丢弃 0–20% 的词，
部分输入加入 today I saw 等口语前缀，目标是完整的原描述。不是用 LLM 合成训练标签。
沿用现有 20000/500/500 张照片对应的 train/val/test 描述，按 image ID 隔离划分。
词表仅来自训练描述；验证集 token NLL 选 best.pt；独立测试集报告 NLL/perplexity。
训练目标是照片描述去噪，尚不是人工标注的“社交帖子→最佳生图提示词”数据。
因此对口语、抽象帖子和复杂新场景存在分布差异，可能重复、漏词和编造内容。
模型最多读取前 39 个词。最终提示词明确保留完整英文原帖为事实依据，模型草稿仅供参考。
解码在重复三词组合时提前结束以避免循环；这是显式解码规则，不是新的训练成果。
最终提示词中的风格指令与事实优先规则是程序拼接，不伪称由模型学习得到。
这版模型是新增研究模块，不能替代保留的 VAE/CNN 在课程中的相关实验。

已有 data/coco/train.json、val.json、test.json 时可直接训练，不需要下载照片或生成 Qwen 特征：
   .\.venv\Scripts\python.exe -m scratch.prompt_train --steps 3000 --minutes 10
续训：
   .\.venv\Scripts\python.exe -m scratch.prompt_train --resume --steps 6000 --minutes 10
评估：
   .\.venv\Scripts\python.exe -m scratch.prompt_train --evaluate
每批原子更新 artifacts/prompt/last.pt，保存模型、优化器和随机状态；best.pt 根据验证集选择。
设置 artifacts/prompt/STOP 文件可在一批结束后暂停，续训前删除此文件。
首次准备数据沿用下文 COCO 准备流程。这个文本模块实际只读取 JSON 描述。
训练权重更换后重启服务。NLL 降低并不证明生成图像更好，需做独立人工对照：
固定同一个 API 模型、质量与风格，比较“原帖直接生图”与“自训提示词生图”。
使用未用于训练/调参的真实帖子，多次生成，并评价人物、动作、地点匹配以及无依据细节。

本机初始结果（2026-11-22，CPU）
1146241 个参数，1200 步；训练描述 100007 条，独立测试描述 2501 条。
第 1200 步验证 NLL 0.907317；测试 NLL 0.846904，perplexity 2.332414。
测试结果见 artifacts/prompt/test_metrics.json，权重不提交 Git，需要另行共享。
英文烧烤场景描述能基本复制保留；口语例句 Today I went to the beach ...
曾输出 the to the beach with friends friends friends.，说明口语改写能力仍然不足。
这版可作为可运行的实验基线，不能声称比直接使用原帖更好。页面保留原帖与草稿供核对。

API 与验证
GET /api/health 返回模型权重是否存在，以及图像 API 是否配置；不会返回密钥。
POST /api/prompt {"post":"..."} 返回真实模型草稿、拼接提示词、训练步数和翻译标记。
POST /api/gpt-image {"prompt":"..."} 返回 PNG；这是唯一外部图片生成入口。
旧 /api/scene 已移除，避免意外启动本机 GPU。原模型、数据和检查点未删除。
POST /api/suggest 和 /api/image 保留原梗图推荐与自训涂鸦功能。
不自动重试付费图片请求，超时后请先检查 API 用量。一次只允许一个本服务的生图请求。
离线检查：.\.venv\Scripts\python.exe -m unittest discover -s tests -v
真实付费 API 调用需用户本机配置有效 key；离线测试使用模拟响应，不代表已完成真实生图。
OpenAI 官方接口说明：https://developers.openai.com/api/reference/resources/images/methods/generate

以下为保留的历史实验与已有课程模块说明。

历史实验分支（暂停，不是当前网页生图路径）：COCO 文本条件场景生成
场景模型直接接收完整文字描述，目标是学习人物、物体、动作和环境的组合。
使用 COCO 2017 人工照片描述。RGB U-Net、语言特征投影及两层条件 Encoder 均随机初始化，
默认用规则允许的预训练 Qwen 提供语言特征。没有使用预训练 CLIP、VAE、视觉骨干或超分模型。
Qwen 既用于描述的语言特征，也用于应用中的中文画面描述整理，不生成图片或图像模型权重。

准备 COCO 数据：
先将官方 annotations_trainval2017.zip 放入 data/coco/。
官方下载页 https://cocodataset.org/#download
有效 HTTPS 标注地址：
https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip
   .\.venv\Scripts\python.exe -m scratch.coco_prepare --train-images 20000 --val-images 500 --test-images 500
下载图像时使用同一官方 S3 存储，保留每张图片的来源与许可 ID。
训练子集取自 train2017。验证与测试是 val2017 中互不重叠的子集，
不是 COCO 官方的无标签 test2017。图片做完整构图等比缩放与灰色补边，不用预训练预处理器。

时间受限的 GPU 训练：
   .\.venv\Scripts\python.exe -m scratch.coco_features --batch-size 64
   .\.venv\Scripts\python.exe -m scratch.coco_train --width 64 --batch-size 32 --steps 100000 --hours 1.5
第一行只运行允许的 Qwen 预计算语言特征，不训练 Qwen；第二行从零训练图像扩散网络。
--text-mode scratch 可做完全自训练语言条件的对照实验，默认 --text-mode qwen。
开发机在同一进程从采样切回训练时出现过 cuDNN 执行错误，
因此本机推荐使用分段启动器，每 1000 批结束进程，再从完整检查点启动下一段：
   .\.venv\Scripts\python.exe -m scratch.run_scene_training --minutes 90 --batch-size 96
--minutes 是累计训练预算，会扣除检查点中已记录的训练时间；达到预算自动停止。
每个 minibatch 仍然保存 last.pt，段末另外保存采样图与验证结果。
本机进一步关闭了融合/Flash 注意力，使用标准 math SDP 和确定性 cuDNN 卷积。
训练与采样串行运行，以减少此 Windows/Blackwell 环境下的 CUDA 执行异常。
--hours 限制本次训练循环时长，存盘、验证与采样可能再耗少量时间。
需要续训时追加 --resume，使用同一数据与模型配置。
场景训练默认 --checkpoint-every 1，每个 minibatch 完成后原子更新 artifacts/scene/last.pt，
保存模型、EMA、AdamW 优化器、步数以及 Python/NumPy/PyTorch CPU/CUDA 随机状态。
每个批次的损失记录在 artifacts/scene/batches.jsonl；每批更新同一个检查点文件。
--save-every 控制验证、best.pt 与采样图的间隔，默认 500 批；可改成 1000。
每批完整存盘会增加 I/O 时间。若以后希望减小写入频率，可显式设置 --checkpoint-every 50。
Ctrl+C 会在当前批次保存后正常退出，并保存一组采样图。
也可创建 artifacts/scene/STOP 文件请求暂停。下次续训前删除这个 STOP 文件。
一次只训练一小段的示例：
   .\.venv\Scripts\python.exe -m scratch.coco_train --width 64 --batch-size 32 --session-batches 1000 --steps 100000 --hours 1.5
接着训练下一段：
   .\.venv\Scripts\python.exe -m scratch.coco_train --resume --session-batches 1000 --steps 100000 --hours 1.5
续训默认沿用检查点中的 batch-size，seed 保持一致。显式传入新的 batch-size 可以调整显存使用，
其后的批次会按新的大小采样，历史权重及优化器会保留。GPU 算子的非确定性可能造成微小数值差异。
该模型只用于实验，初次 1–2 小时训练不能保证通用、写实或正确遵循复杂描述。
原生图片为 64×64，界面放大显示不会增加真实细节。
训练产物在 artifacts/scene/，EMA 模型用于采样。
定期采样的四个提示词保存在 sample_prompts.json，使用相同初始噪声对比文字作用。
早期快速 DDIM 采样出现明显噪声伪影，现默认使用完整 1000 步 DDPM 采样。
随机采样图逐图记录 sampler 元数据；生成耗时会比原快速采样长。
完成训练后：
   .\.venv\Scripts\python.exe -m scratch.coco_evaluate
测试集记录配对/错配文字的去噪误差和固定噪声的文字敏感度。
这些指标不能证明图片美观或语义正确，需同时人工检查采样图。

上述命令仅用于记录历史从零扩散实验。目前训练已暂停，不应直接运行这些 GPU 命令。
当前网页使用本文件开头说明的 GPT Image 路径；涂鸦继续使用下面描述的 VAE/CNN。

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
当前版本固定使用 CPU；此前的 POSTPALETTE_LLM_DEVICE=cuda 设置不再生效。
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
- LLM：Qwen/Qwen2.5-1.5B-Instruct，是本机唯一加载外部预训练权重的模型。
  处理中文翻译、候选梗图排序、涂鸦类别选择和短配文。
  新版场景图片另外调用外部预训练 GPT Image API，必须明确披露这部分使用情况。
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
分享源码时排除 .env、.venv、.model-cache、data、artifacts 和 __pycache__。
团队复现需要运行数据准备与训练，或单独共享本组训练产物及其 manifest/history。
旧版 .model-cache 中的其他预训练权重不会被当前代码使用。
