PostPalette · 网络梗图推荐、自训提示词与 GPT 场景配图

根据一段社交帖子推荐 Imgflip 梗图，生成可编辑的英文绘图提示词，再调用 GPT Image 生成场景配图；
也可以用自训 VAE/CNN 生成六类涂鸦贴纸。服务只监听本机 127.0.0.1:8765，附带 Chrome 扩展入口。


======================================================================
一、当前用法
======================================================================

1. 安装（首次）
----------------------------------------------------------------------
开发机使用 Python 3.12；macOS 上用 Python 3.14 也已验证可以安装并通过离线测试。
本地推理全部固定在 CPU 上，不需要 GPU，也不需要安装 CUDA 版 PyTorch。

macOS / Linux（终端）：
   cd PostPalette
   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt

Windows（PowerShell）：
   cd PostPalette
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt

首次推荐梗图或翻译中文时，会从 Hugging Face 下载 Qwen/Qwen2.5-1.5B-Instruct（约 3 GB），
缓存在 .model-cache/。

2. 放入自训模型权重
----------------------------------------------------------------------
自训权重不提交 Git，也没有下载脚本，需要向组内负责训练的同学索取，按下面的路径放好：

   artifacts/emotion/best.pt   情绪识别 Transformer Encoder   由 python -m scratch.train emotion 产出
   artifacts/cnn/best.pt       涂鸦筛选 CNN                   由 python -m scratch.train cnn 产出
   artifacts/vae/best.pt       涂鸦生成条件 VAE               由 python -m scratch.train vae 产出
   artifacts/prompt/best.pt    提示词草稿 GRU 编码器-解码器    由 python -m scratch.prompt_train 产出

加载时会校验权重里记录的任务名、随机初始化标记、训练步数、标签顺序和词表，
只接受本项目训练脚本产出的文件。放入或更换权重后需要重启服务。

各功能需要的条件：
   推荐梗图                   只需 Qwen；有 emotion 权重时额外参考情绪，缺少时自动降级为只用 Qwen
   根据帖子生成提示词          prompt 权重（中文帖子还需要 Qwen 翻译）
   GPT 场景配图               OPENAI_API_KEY（见第四节）
   涂鸦贴纸                   vae 和 cnn 两个权重

3. 配置 GPT Image（可选）
----------------------------------------------------------------------
macOS / Linux：   cp .env.example .env
Windows：         Copy-Item .env.example .env
在本机编辑 .env，填写 OPENAI_API_KEY。详细说明和验证步骤见第四节。
不配置也能启动服务，只是“GPT 场景配图”的生成按钮会被禁用。

4. 启动
----------------------------------------------------------------------
macOS / Linux：   .venv/bin/python server.py
Windows：         .\.venv\Scripts\python.exe server.py

浏览器打开 http://127.0.0.1:8765/ ，Ctrl+C 停止服务。可用 --port 指定其他端口，
但 Chrome 扩展固定连接 8765。

5. 页面状态说明
----------------------------------------------------------------------
页面启动时读取 GET /api/health，按结果直接禁用不可用的功能，并在按钮下方写明原因：
   - 右上角徽标显示“本地服务已连接”“部分模型待训练”“缺少模型依赖”或“请启动本地服务”。
     缺少任一自训权重时显示“部分模型待训练”。
   - 缺 prompt 权重：“根据帖子生成提示词”禁用。
   - 缺 vae 或 cnn 权重：涂鸦模式下“生成图片”禁用。
   - 未配置 OPENAI_API_KEY：GPT 场景配图模式下“生成图片”禁用。
   - 缺 emotion 权重：“使用情绪模型辅助推荐”自动取消勾选并禁用，推荐照常可用。
推荐结果是降级结果时（未使用情绪模型、Qwen 没给出有效涂鸦类别），状态栏用黄色文字列出原因。

6. 使用流程
----------------------------------------------------------------------
梗图推荐：输入帖子或点“试用示例”，点“推荐梗图”。返回三张 Imgflip 梗图、一个涂鸦类别和英文配文。
场景配图：
   1) 生成方式选“GPT 场景配图”，点“根据帖子生成提示词”。不调用付费 API。
      中文帖子先由本地 Qwen 翻译，可能较慢；英文直接进入自训 GRU。
   2) 检查模型草稿和可编辑提示词。也可以手动输入提示词作对照，手动内容不计作自训模型输出。
   3) 点“生成图片”。此时才把当前提示词发送到 OpenAI，按用量计费；输出 1024×1024 PNG。
涂鸦贴纸：生成方式选“涂鸦贴纸”，选择类别、颜色、种子和配图/透明贴纸，生成并下载 PNG。
   VAE 输出 28×28，放大排版为 512×512。配文单独复制，不写进 PNG。

7. Chrome 扩展
----------------------------------------------------------------------
保持服务运行在默认端口 8765。在 chrome://extensions/ 开启开发者模式，
点击“加载已解压的扩展程序”，选择 extension/ 文件夹。
在普通网页选中帖子文字，再点 PostPalette 图标，工作台会带入选中文字。
浏览器设置页等特殊页面不允许扩展读取选区，可直接在工作台粘贴文字。

8. 离线测试
----------------------------------------------------------------------
macOS / Linux：   .venv/bin/python -m unittest discover -s tests -v
Windows：         .\.venv\Scripts\python.exe -m unittest discover -s tests -v
测试全部离线运行，自训模型、Qwen、Imgflip 和 OpenAI 调用都用模拟对象代替，不需要权重和 API key。
   tests/test_models.py          suggest / generate_prompt / generate_image 的正常、缺权重降级、非法类别等路径
   tests/test_routes.py          /api/image、/api/suggest、/api/prompt 的 HTTP 行为和输入校验
   tests/test_image_pipeline.py  GPT Image 接入、日志不含密钥、路由与 Origin 校验
GitHub Actions（.github/workflows/tests.yml）在每次 push 和 pull request 时运行同一套测试，
安装 CPU 版 PyTorch，不下载权重，也不调用任何外部 API。
通过测试不代表真实生图已经验证，见第四节。


======================================================================
二、接口
======================================================================

GET  /api/health      依赖是否齐全、四个权重是否存在（checkpoints）、图像 API 是否配置；不返回密钥。
POST /api/suggest     {"post": "...", "use_sentiment": true}
                      返回 memes、category（可能为 null）、caption、sentiment，以及
                      emotion_requested、emotion_used、degraded、notices（降级原因）。
POST /api/prompt      {"post": "..."}  返回模型草稿、拼接后的提示词、训练步数和是否翻译。
POST /api/gpt-image   {"prompt": "..."}  返回 PNG。这是唯一的外部图片生成入口。
POST /api/image       {"category", "color", "mode", "seed"}  返回自训涂鸦 PNG。
只接受来自本地工作台的请求（校验 Host 和 Origin）。
付费图片请求不自动重试；服务一次只处理一个生图请求。
模型或外部服务失败时返回 503 和中文说明，终端记录诊断信息（图像 API 的日志会去除密钥）。


======================================================================
三、模块与课程主题
======================================================================

当前版本采用“预训练 LLM + 四个从零训练的模型 + 外部图像 API”。
准备代码和数据不等于模型已训练，可通过 /api/health 确认权重是否存在。

- Transformer Encoder（情绪识别）：随机初始化的两层 Encoder + 训练集词表及随机词向量，
  在 GoEmotions 合并后的八类情绪上训练，采用多标签 BCE 损失。
- 提示词草稿（scratch/prompt_model.py）：随机词向量 + GRU Encoder/Decoder + 点积注意力 + 复制概率混合。
  没有加载预训练文本编码器、词向量或其他非 LLM 权重。
- Traditional GenAI（涂鸦）：随机初始化条件 VAE，在 QuickDraw 六类图像上训练，
  用 BCE 重建损失与 KL 正则学习可按类别采样的潜空间。
- CNN（涂鸦筛选）：独立随机初始化的卷积分类器，识别六类涂鸦。VAE 每次生成 16 个候选，
  CNN 按目标类别分数排序选出一个。与 VAE 共用固定的数据划分。
- LLM：Qwen/Qwen2.5-1.5B-Instruct，是本机唯一加载外部预训练权重的模型。
  处理中文翻译、候选梗图排序、涂鸦类别选择和短配文。
- 外部图像 API：场景配图调用 OpenAI GPT Image，必须明确披露这部分使用情况。

课程边界：按小组当前选择接入 GPT 生图；课件列出 ChatGPT 等 LLM，但未明确图像能力的例外。
因此本 README 不宣称教授已批准外部生图，也不把 API 输出计作从零训练的图像模型成果。

提示词模型的真实训练内容
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

梗图来源
https://imgflip.com/api
从公开热门模板目录读取图片地址和热度，用途标注和关键词筛选后由 Qwen 排序。
目录请求失败时用内置模板元数据，缩略图仍需联网。不向 Imgflip 发送帖子内容。
图片显示来源链接，数据集许可不等同于第三方梗图图片许可。


======================================================================
四、GPT Image 验证（待组员完成）
======================================================================

目前 GPT Image 只做过离线模拟测试，还没有用真实 key 调用过。请有 API 额度的组员按下面步骤验证一次。

1. 配置项
----------------------------------------------------------------------
三个变量，写在项目根目录的 .env 文件里（从 .env.example 复制）：
   OPENAI_API_KEY=sk-...                 必填
   POSTPALETTE_IMAGE_MODEL=gpt-image-1.5 可选，默认 gpt-image-1.5，必须以 gpt-image- 开头
   POSTPALETTE_IMAGE_QUALITY=medium      可选，low / medium / high / auto，默认 medium
服务只从 .env 读取这三个变量名，其他行会被忽略。
如果同名系统环境变量已经存在，以系统环境变量为准，.env 里的值不会覆盖它。
修改后需要重启服务。

.env 已被 .gitignore 忽略（只有 .env.example 会进 Git）。不要提交 .env，
不要把 key 发到聊天、截图或组员压缩包里，也不要填进网页。
提交前可用 git status 确认 .env 没有出现在待提交列表里。

2. 确认 key 已被读到
----------------------------------------------------------------------
启动服务后运行：
   curl -s http://127.0.0.1:8765/api/health
返回内容中应有 "image_api": {"configured": true, "model": "...", "quality": "..."}。
接口只返回“是否已配置”，不会返回 key 本身。
页面上“GPT 场景配图”区域也会显示“图像 API 已配置 · 模型名 · 质量”，生成按钮变为可用。

3. 最小成本的真实调用
----------------------------------------------------------------------
API 请求按用量计费，与 ChatGPT 订阅额度分开。为了把成本降到最低：
   - 在 .env 中设置 POSTPALETTE_IMAGE_QUALITY=low 后重启服务（尺寸固定为 1024×1024）。
   - 只发一次请求，用一句简短的手写提示词，不必先跑 Qwen 或自训模型。

macOS / Linux：
   curl -s -X POST http://127.0.0.1:8765/api/gpt-image \
        -H "Content-Type: application/json" \
        -d '{"prompt":"a red apple on a white table"}' -o gpt-test.png
   file gpt-test.png        # 成功时显示 PNG image data, 1024 x 1024
Windows（PowerShell）：
   Invoke-WebRequest -Method Post -Uri http://127.0.0.1:8765/api/gpt-image `
        -ContentType "application/json" -Body '{"prompt":"a red apple on a white table"}' -OutFile gpt-test.png
也可以直接在网页里手动输入提示词，点一次“生成图片”。

失败时响应是一段 JSON 中文说明，启动服务的终端会打印一行诊断信息，例如：
   Inference failed (/api/gpt-image): OpenAI HTTP 403 request_id=req_... type=... code=... message=...
日志里的 key 和 sk- 开头的片段已被替换为 [REDACTED]。
超时后请先到 OpenAI 用量页确认请求是否已被处理，再决定是否重试，避免重复计费。

4. 模型无权限时的表现和备选
----------------------------------------------------------------------
gpt-image-1.5 不可用时，页面会显示以下之一：
   403  “此 API 项目暂无图像模型访问权限，请检查账号权限或组织验证。”
        GPT Image 系列模型通常要求 OpenAI 组织先完成身份验证（Organization verification）。
   404  “图像模型不可用，请检查配置的模型名。”
   400  “图像请求被拒绝，请调整描述或检查模型配置。”（模型名或参数不被接受时也可能是 400）
具体原因看终端日志里的 message 字段。
备选：把 .env 中的 POSTPALETTE_IMAGE_MODEL 改为 gpt-image-1 后重启服务再试一次。
如果账号的模型列表里有更便宜的 gpt-image-1-mini，也可以使用。
请以 OpenAI 控制台当前列出的可用模型为准。

5. 回报结果
----------------------------------------------------------------------
验证完成后，请把结果发给【负责人：待填写】，内容包括：
   - 日期、使用的模型名和质量
   - 成功或失败；失败时附上终端里那一行 Inference failed 日志（已脱敏，可以直接贴）
   - 成功时附上生成的 gpt-test.png，以及 OpenAI 用量页显示的这次费用
不要在回报里附带 .env 或 key。


======================================================================
五、训练与数据（参考，应用运行不需要）
======================================================================

以下命令会下载数据或训练模型，只在需要重新产出 artifacts/ 权重时使用。
命令以 Windows 写法给出；macOS / Linux 把 .\.venv\Scripts\python.exe 换成 .venv/bin/python。

情绪 / CNN / VAE
----------------------------------------------------------------------
仅准备数据和检查，不训练：
   .\.venv\Scripts\python.exe -m scratch.prepare
   .\.venv\Scripts\python.exe -m scratch.check
prepare 不导入 PyTorch，不加载模型。check 只做 CPU 前向与数据检查，不做反向传播或更新权重。
准备脚本会下载 GoEmotions 和六类 QuickDraw，每类先取 8000 个样本。
数据位于 data/，训练产物位于 artifacts/，均不提交 Git。
数据已准备好时直接运行 check；再次 prepare 会按相同规则重建数据。

下面的命令会真正开始训练：
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

训练后单独评估（这些命令不更新权重）：
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

提示词 GRU
----------------------------------------------------------------------
训练数据 data/coco/{train,val,test}.json 由 COCO 准备脚本生成（该脚本已随 COCO 实验移到
scratch/experiments/，见第六节）。已有这三个 JSON 时可直接训练，不需要下载照片或生成 Qwen 特征：
   .\.venv\Scripts\python.exe -m scratch.prompt_train --steps 3000 --minutes 10
续训：
   .\.venv\Scripts\python.exe -m scratch.prompt_train --resume --steps 6000 --minutes 10
评估：
   .\.venv\Scripts\python.exe -m scratch.prompt_train --evaluate
每批原子更新 artifacts/prompt/last.pt，保存模型、优化器和随机状态；best.pt 根据验证集选择。
设置 artifacts/prompt/STOP 文件可在一批结束后暂停，续训前删除此文件。
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

数据来源与处理记录
----------------------------------------------------------------------
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

分享项目
----------------------------------------------------------------------
分享源码时排除 .env、.venv、.model-cache、data、artifacts 和 __pycache__。
团队复现需要运行数据准备与训练，或单独共享本组训练产物及其 manifest/history。
旧版 .model-cache 中的其他预训练权重不会被当前代码使用。


======================================================================
六、历史实验（已暂停）：COCO 文本条件场景扩散
======================================================================

这部分已暂停，不是当前网页的生图路径，主线代码不调用它。代码位于 scratch/experiments/，
暂停原因见 scratch/experiments/README.md。原 /api/scene 接口已删除，原模型、数据和检查点未删除。
所有模块路径已从 scratch.coco_* 改为 scratch.experiments.coco_*。

场景模型直接接收完整文字描述，目标是学习人物、物体、动作和环境的组合。
使用 COCO 2017 人工照片描述。RGB U-Net、语言特征投影及两层条件 Encoder 均随机初始化，
默认用规则允许的预训练 Qwen 提供语言特征。没有使用预训练 CLIP、VAE、视觉骨干或超分模型。
Qwen 只提供描述的语言特征，不生成图片或图像模型权重。

准备 COCO 数据（也是提示词 GRU 的训练数据来源）：
先将官方 annotations_trainval2017.zip 放入 data/coco/。
官方下载页 https://cocodataset.org/#download
有效 HTTPS 标注地址：
https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip
   .\.venv\Scripts\python.exe -m scratch.experiments.coco_prepare --train-images 20000 --val-images 500 --test-images 500
下载图像时使用同一官方 S3 存储，保留每张图片的来源与许可 ID。
训练子集取自 train2017。验证与测试是 val2017 中互不重叠的子集，
不是 COCO 官方的无标签 test2017。图片做完整构图等比缩放与灰色补边，不用预训练预处理器。

时间受限的 GPU 训练：
   .\.venv\Scripts\python.exe -m scratch.experiments.coco_features --batch-size 64
   .\.venv\Scripts\python.exe -m scratch.experiments.coco_train --width 64 --batch-size 32 --steps 100000 --hours 1.5
第一行只运行允许的 Qwen 预计算语言特征，不训练 Qwen；第二行从零训练图像扩散网络。
--text-mode scratch 可做完全自训练语言条件的对照实验，默认 --text-mode qwen。
开发机在同一进程从采样切回训练时出现过 cuDNN 执行错误，
因此本机推荐使用分段启动器，每 1000 批结束进程，再从完整检查点启动下一段：
   .\.venv\Scripts\python.exe -m scratch.experiments.run_scene_training --minutes 90 --batch-size 96
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
   .\.venv\Scripts\python.exe -m scratch.experiments.coco_train --width 64 --batch-size 32 --session-batches 1000 --steps 100000 --hours 1.5
接着训练下一段：
   .\.venv\Scripts\python.exe -m scratch.experiments.coco_train --resume --session-batches 1000 --steps 100000 --hours 1.5
续训默认沿用检查点中的 batch-size，seed 保持一致。显式传入新的 batch-size 可以调整显存使用，
其后的批次会按新的大小采样，历史权重及优化器会保留。GPU 算子的非确定性可能造成微小数值差异。
该模型只用于实验，初次 1–2 小时训练不能保证通用、写实或正确遵循复杂描述。
原生图片为 64×64，界面放大显示不会增加真实细节。
训练产物在 artifacts/scene/，EMA 模型用于采样。
定期采样的四个提示词保存在 sample_prompts.json，使用相同初始噪声对比文字作用。
早期快速 DDIM 采样出现明显噪声伪影，现默认使用完整 1000 步 DDPM 采样。
随机采样图逐图记录 sampler 元数据；生成耗时会比原快速采样长。
完成训练后：
   .\.venv\Scripts\python.exe -m scratch.experiments.coco_evaluate
测试集记录配对/错配文字的去噪误差和固定噪声的文字敏感度。
这些指标不能证明图片美观或语义正确，需同时人工检查采样图。
