PostPalette · 网络梗图推荐、自训提示词与 GPT 场景配图

根据一段社交帖子推荐 Imgflip 梗图（可换一批），生成可编辑的英文绘图提示词，再调用 GPT Image 生成场景配图；
也可以用本地 Qwen 改写帖子、检索相似参考，从照片出发写推文，或用自训 VAE/GAN + CNN 生成六类涂鸦贴纸。服务只监听本机 127.0.0.1:8765，附带 Chrome 扩展入口。


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

2. 下载本组已训练模型（models-v2，无需重新训练）
----------------------------------------------------------------------
GitHub 源码不包含 artifacts/，这是刻意排除的大文件，不是训练成果缺失；git clone 或下载源码 ZIP
都不会自动下载权重。当前功能的全部自训模型通过 GitHub Release 单独提供：
   发布页：https://github.com/TiantongZhang1/PostPalette/releases/tag/models-v2
   模型包：https://github.com/TiantongZhang1/PostPalette/releases/download/models-v2/postpalette-models-v2.zip
   校验值：https://github.com/TiantongZhang1/PostPalette/releases/download/models-v2/postpalette-models-v2.zip.sha256
v2 ZIP SHA-256：bcc09253a733ca089740cebe9908d8eda996e054517e88b044442d3883b9c7f7

把 ZIP 下载到包含 server.py 的项目根目录。已经有自己训练的同名权重时先备份，下面的命令会覆盖。
只解压 artifacts/ 这一部分，包内的 MODEL_INSTALL.txt 和 MODEL_MANIFEST.json 不会落到项目根目录：

macOS / Linux：
   shasum -a 256 postpalette-models-v2.zip      # 输出应与上面的 SHA-256 一致
   unzip -o postpalette-models-v2.zip 'artifacts/*' -d .
（.sha256 文件是 Windows 换行格式，macOS 上 shasum -c 会报找不到文件，请直接对比输出的哈希值。）

Windows（PowerShell）：
   (Get-FileHash .\postpalette-models-v2.zip -Algorithm SHA256).Hash   # 应与上面的 SHA-256 一致（大小写不敏感）
   Expand-Archive -LiteralPath .\postpalette-models-v2.zip -DestinationPath .\models-v2-tmp -Force
   Copy-Item -Path .\models-v2-tmp\artifacts -Destination . -Recurse -Force
   Remove-Item -Recurse -Force .\models-v2-tmp

解压后得到 9 个当前模型，路径为 artifacts/<任务>/best.pt，不要再套一层 artifacts/：
   emotion          情绪识别 Transformer Encoder          python -m scratch.train emotion
   cnn              涂鸦筛选 CNN                          python -m scratch.train cnn
   vae              涂鸦生成条件 VAE                      python -m scratch.train vae
   gan              涂鸦生成条件 GAN                      python -m scratch.train gan
   prompt           提示词草稿 GRU 编码器-解码器           python -m scratch.prompt_train
   photo_tags       照片 80 类物体标签 CNN                python -m scratch.photo_train
   person_pose      人物候选框与 17 点关键点               python -m scratch.person_train
   expression       可见表情 CNN                          python -m scratch.people_train expression
   daily_activity   整图日常活动 CNN（含 calibration.json）python -m scratch.activity_train
包内另有训练记录、测试指标、日常活动选模记录、VAE/GAN 样例和逐文件 SHA-256 清单。
旧 action 模型与暂停的 scene 模型在同页 postpalette-experiments-v2.zip，可选下载，不是当前默认功能。
只发布原始 best.pt（原检查点已有的优化器/元数据保留）；从最新批次续训还需本地 last.pt 与相同的数据划分。
旧版 models-v1 只含前 4 个涂鸦/文本模型（不含 gan 和照片相关模型），不适用于本版全部功能。
ZIP、.sha256、.manifest.json、MODEL_INSTALL.txt 和 MODEL_MANIFEST.json 已加入 .gitignore，
误放在根目录也不会被提交。

加载时会校验权重里记录的任务名、随机初始化标记、训练步数、标签顺序和词表，
只接受本项目训练脚本产出的文件。放入或更换权重后需要重启服务，以清除进程内的模型缓存。
模型包不含 Qwen 权重（首次使用时自动下载）、GPT 生图 API 密钥、原始数据、本机缓存或用户上传的图片。
模型指标和质量限制见第三、五节和包内说明；尤其不能把 CNN 分类准确率当作贴纸生成质量。

各功能需要的条件：
   推荐梗图                   只需 Qwen；有 emotion 权重时额外参考情绪，缺少时自动降级为只用 Qwen
   根据帖子生成提示词          prompt 权重（中文帖子还需要 Qwen 翻译）
   GPT 场景配图               OPENAI_API_KEY（见第四节）
   涂鸦贴纸                   cnn，以及所选画法对应的 vae 或 gan 权重
   帖子改写                   只需 Qwen
   找相似参考                 不需要模型（TF-IDF 词面检索）
   照片辅助写推文             photo_tags、daily_activity、person_pose、expression；某个模型不可用时其余结果照常显示

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
   - 缺 cnn 或所选画法（VAE / GAN）的权重：涂鸦模式下“生成图片”禁用；切换画法时重新判断。
   - 未配置 OPENAI_API_KEY：GPT 场景配图模式下“生成图片”禁用。
   - 缺 emotion 权重：“使用情绪模型辅助推荐”自动取消勾选并禁用，推荐照常可用。
推荐结果是降级结果时（未使用情绪模型、Qwen 没给出有效涂鸦类别），状态栏用黄色文字列出原因。
照片区域的各个模型由照片流程自己根据 /api/health 显示是否可用，见第三节“照片流程整合”。

6. 使用流程
----------------------------------------------------------------------
梗图推荐：输入帖子或点“试用示例”，点“推荐梗图”。返回三张 Imgflip 梗图、一个涂鸦类别和英文配文；
   “换一批梗图”会避开本轮已经展示过的模板。详见第三节“梗图库扩展”。
完善帖子：在“完善帖子”中选择扩写、幽默或简洁，生成 3 个版本；“找相似参考”检索内置示例。
从照片开始：展开“从照片开始写推文”，上传照片 → 分析 → 核对线索 → 统一确认 → 三个可编辑草稿。
场景配图：
   1) 生成方式选“GPT 场景配图”，点“根据帖子生成提示词”。不调用付费 API。
      中文帖子先由本地 Qwen 翻译，可能较慢；英文直接进入自训 GRU。
   2) 检查模型草稿和可编辑提示词。也可以手动输入提示词作对照，手动内容不计作自训模型输出。
   3) 点“生成图片”。此时才把当前提示词发送到 OpenAI，按用量计费；输出 1024×1024 PNG。
涂鸦贴纸：生成方式选“涂鸦贴纸”，“生成模型”可选条件 VAE 或条件 GAN，再选择类别、颜色、种子和
   配图/透明贴纸，生成并下载 PNG。两种画法都生成 16 个候选，由 CNN 按目标类别分数选一张；
   同一个种子在同一种画法下可复现，换画法会得到不同的图。
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
   tests/test_gan.py、test_memes.py、test_writing.py、test_photo_*.py、test_people.py、test_daily_activity.py
                                 GAN、梗图库、改写与参考检索、照片与人物分析等功能
GitHub Actions（.github/workflows/tests.yml）在每次 push 和 pull request 时运行同一套测试，
安装 CPU 版 PyTorch，不下载权重，也不调用任何外部 API。
通过测试不代表真实生图已经验证，见第四节。


======================================================================
二、接口
======================================================================

GET  /api/health      依赖是否齐全、涂鸦/文本权重是否存在（checkpoints）、图像 API 是否配置，
                      以及 photo_tags、people_models、daily_activity 的可用状态；不返回密钥。
POST /api/suggest     {"post": "...", "use_sentiment": true}
                      可选 "exclude_ids"（最多 120 个模板 ID，用于换一批）。
                      返回 memes（含模板 id）、category（可能为 null）、caption、sentiment、catalog（图库状态），以及
                      emotion_requested、emotion_used、degraded、notices（降级原因）。
POST /api/prompt      {"post": "..."}  返回模型草稿、拼接后的提示词、训练步数和是否翻译。
POST /api/gpt-image   {"prompt": "..."}  返回 PNG。这是唯一的外部图片生成入口。
POST /api/image       {"category", "color", "mode", "seed", "generator": "vae"|"gan"}  返回自训涂鸦 PNG。
POST /api/rewrite     {"post": "...", "style": "expand|humor|concise"}  返回 Qwen 改写的三个版本。
POST /api/references  {"post": "..."}  返回前三条词面匹配的参考示例；不依赖模型、不联网。
POST /api/photo-analyze  照片原始二进制（image/jpeg|png|webp，≤8 MB），一次返回活动、物体、人物分析。
POST /api/photo-tags、/api/photo-activity、/api/photo-people  同上，单独返回某一项（保留给旧客户端和测试）。
POST /api/photo-post  {"labels", "context", "style", "confirmed": true, "activity", "activity_confirmed"}
                      根据用户确认过的文字线索生成三个推文草稿，不接收图像。
只接受来自本地工作台的请求（校验 Host 和 Origin）。
付费图片请求不自动重试；服务一次只处理一个生图请求。
模型或外部服务失败时返回 503 和中文说明，终端记录诊断信息（图像 API 的日志会去除密钥）。


======================================================================
三、模块与课程主题
======================================================================

当前版本采用“预训练 LLM（Qwen）+ 本组从零训练的模型 + 外部图像 API”。
文本与涂鸦部分有五个自训模型：情绪 Encoder、CNN、条件 VAE、条件 GAN 和提示词 GRU；
照片部分另有物体标签、人物关键点、可见表情和日常活动四个自训模型（见下文“照片相关模型”）。
准备代码和数据不等于模型已训练，可通过 /api/health 确认权重是否存在。

- Transformer Encoder（情绪识别）：随机初始化的两层 Encoder + 训练集词表及随机词向量，
  在 GoEmotions 合并后的八类情绪上训练，采用多标签 BCE 损失。
- 提示词草稿（scratch/prompt_model.py）：随机词向量 + GRU Encoder/Decoder + 点积注意力 + 复制概率混合。
  没有加载预训练文本编码器、词向量或其他非 LLM 权重。
- Traditional GenAI（VAE）：随机初始化条件 VAE，在 QuickDraw 六类图像上训练，
  用 BCE 重建损失与 KL 正则学习可按类别采样的潜空间。
- Traditional GenAI（GAN）：随机初始化条件 GAN（ACGAN），在同一份 QuickDraw 六类数据
  和同一份划分上训练。判别器有真假与六类两个输出头，生成器用翻转标签的对抗损失加分类损失。
  这是网页上可选的第二种涂鸦画法，不是用来和 VAE 比较优劣的对照实验。
- CNN（涂鸦筛选）：独立随机初始化的卷积分类器，识别六类涂鸦。VAE 与 GAN 每次都生成 16 个候选，
  CNN 按目标类别分数排序选出一个。三者共用固定的数据划分。
- LLM：Qwen/Qwen2.5-1.5B-Instruct，是本机唯一加载外部预训练权重的模型。
  处理中文翻译、候选梗图排序、涂鸦类别选择、短配文、帖子改写和照片线索写作。
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
完整公开热门模板目录、历史缓存及内置快照合并参与推荐，用途标注和主题词检索后由 Qwen 排序。
目录请求失败时使用上次缓存或内置快照，缩略图仍需联网。不向 Imgflip 发送帖子内容。
图片显示来源链接，数据集许可不等同于第三方梗图图片许可。


以下功能说明来自组员的开发记录（2026-11-25～27），保留原文。其中的命令按 Windows 写法给出，macOS / Linux 把 .\.venv\Scripts\python.exe 换成 .venv/bin/python。

梗图库扩展（2026-11-27）
旧版仅从 22 张模板中筛出 8 张。本版让完整 Imgflip 热门目录进入推荐，
随项目附带本次获取的 100 张模板元数据（content/meme_catalog.json），加上用途与关键词标注。
每次联网获取当前热门目录，成功后在内存缓存 10 分钟；失败后 1 分钟再试。
历史热门目录保存在 .model-cache/memes/catalog.json，与内置目录合并去重，最多保留 500 张。
断网依次使用上次成功缓存、随项目附带目录，最后才退回基础 22 张。图片本身仍需联网。
先做词面/主题扩展检索，保留 18 张相关候选，加 6 张探索候选交给本地 Qwen 排序，返回 3 张。
探索来自下一档候选，不保证每张图片都相关；最终仍需用户判断，尚无人工相关性评测指标。
“换一批梗图”通过模板 ID 排除本轮已看/已排除的模板；候选不足 3 张时提示开启新一轮。
浏览器仅保存最近 18 个模板 ID，后续推荐自动避开；不保存帖子内容，可清除网站存储重置。
POST /api/suggest 新增可选 exclude_ids（最多 120 个数字字符串 ID），返回 catalog 状态与每张模板 ID。
页面显示图库数量和在线/缓存状态；换一批不会改掉已编辑配文或已生成图片。
这是热门模板检索，不是全网图片搜索；当前来源以欧美梗图为主，未接入中文梗图库。
未使用新的预训练视觉模型，无需重新训练。公共目录请求不包含用户帖子。

照片流程整合（2026-11-27，当前界面）
现在统一为：添加照片 → 分析照片 → 核对写作线索 → 统一确认 → 三个可编辑草稿。
一个“分析照片”按钮调用 /api/photo-analyze，一次上传后在本机依次分析活动、物体、人物。
每个模块独立处理失败；某项模型不可用时，其余结果保留，仍可手动补充说明继续写作。
主要活动与已选物体集中展示；80 个完整物体标签收进“添加或修改物体”，可以搜索、增删。
候选分数、人物框、姿态与可见表情收进“查看识别详情与人物候选”。人物描述默认不用于写作，
用户检查后可点击加入同一个补充说明框，也可直接填写自己确认的人数、地点、活动等内容。
全部线索只需统一确认一次。修改活动、物体或补充说明后，旧确认和旧照片草稿自动清除。
只填写人工说明也可以写作，不需要为了通过表单而强选不确定的活动或物体。
照片草稿显示在照片区域内，可编辑、复制或“使用此版本”接入原帖、梗图推荐与配图流程。
此处整合了操作与接口，没有重新训练模型；原模型效果、数据来源和课程边界保持原记录。
旧 /api/photo-tags、/api/photo-activity、/api/photo-people 接口保留，供独立测试与旧客户端使用。
新的照片界面脚本为 dist/photo.js，主应用在 dist/app.js，均无需额外构建即可由 server.py 提供。

帖子扩写与参考检索（2026-11-25）
在帖子输入框下的“完善帖子”选择扩写、幽默或简洁，点击“生成 3 个版本”。
本地 Qwen 直接处理中文/英文，在 CPU 上返回三个可编辑版本；不调用付费 API、不新增训练。
提示模型保留事实，但这不是事实一致性保证，必须检查人物、地点、数字等细节。
点击“使用此版本”替换当前帖子，并清除旧推荐/图片/提示词，避免旧结果与新文本不对应。
“恢复替换前原文”只保存本次页面内最近一次替换前的内容，不会把帖子写入磁盘。

“找相似参考”检索 content/reference_posts.json 中的 24 条 AI 编写示例。
这些是项目示例，不是真实用户帖子，不来自互联网搜索，不宣称是组员原创数据。
采用中日韩统一汉字区中文连续二/三字片段与英文单词的 TF-IDF 余弦相似度。
中文与英文关键词共同索引，属于词面检索；未使用预训练 Embedding、Sentence-BERT 或自训语义模型。
页面显示匹配词、来源说明和分数；没有足够词面匹配时返回空结果，不补造来源。
这是流程验证用的小型检索基线，尚不代表开放场景语义检索质量。
可在 JSON 中加入经过授权的参考文本，维护唯一 id、topic、text、keywords 及相应来源信息，重启服务更新索引。
目前文件内统一的示例来源标签仅适用于 AI 示例；接入其他来源前需同步调整逐条来源字段和展示逻辑。
POST /api/rewrite {"post":"...","style":"expand|humor|concise"} 返回三个版本和模型说明。
POST /api/references {"post":"..."} 返回前三条词面匹配及参考库说明；此接口无模型依赖、不联网。
课程对应：扩写是允许的 LLM 应用；TF-IDF 为检索基线，不计作新增神经网络课程主题。


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
GAN 需要先训练好 CNN，因为它按验证集上 CNN 认出目标类别的比例来挑选检查点：
   .\.venv\Scripts\python.exe -m scratch.train gan --epochs 40
建议依次执行。emotion/cnn/vae 的默认设备为 CUDA，CPU 训练可添加 --device cpu。
gan 的默认设备是 CPU，模型只有约 73 万参数，本机一轮约 18 秒，不需要 GPU。
gan 的默认学习率是 2e-4（DCGAN/ACGAN 常用值），与其他三个任务的 0.001 不同；
显式传入 --lr 可覆盖，检查点记录的是本次实际使用的值。
gan 不支持 --evaluate：生成器没有有意义的测试集损失，不要为了形式统一制造一个会被误引用的指标。
改看 artifacts/gan/samples.png，以及 artifacts/gan/history.json 里逐轮的验证可识别率。
该可识别率只说明 CNN 是否认得出这一类，不衡量美观，也不代表 GAN 与 VAE 孰优孰劣；
GAN 按这个指标选检查点，VAE 按验证损失选，两者的选模口径本就不同。
CNN 在 GAN 训练中只做前向，不参与生成器的反向传播。
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

2026-11-25 条件 GAN 首轮训练结果（CPU）
729096 个可训练参数（生成器侧 552897，判别器侧 176199），40 轮共约 723 秒，
平均每轮约 18 秒，batch size 128，lr 2e-4，seed 2026。训练分两段完成，中途续训一次。
验证集可识别率第 1 轮 0.4192，第 10 轮 0.8104，第 16 轮达到最高 0.8335，
此后在 0.7840–0.8290 之间波动，第 40 轮为 0.7840。best.pt 因此取自第 16 轮。
可识别率是用已训练的 CNN 在验证集上算的，衡量 CNN 能否认出目标类别，
不衡量美观，也不构成与 VAE 的优劣比较——两者的选模口径本就不同。
GAN 的损失与生成质量没有单调关系，所以选检查点用的是这个指标而不是损失。
采样图位于 artifacts/gan/samples.png，六行分别对应六个类别，每行内部有明显差异，
没有出现模式崩溃；猫耳、杯柄、花茎等结构基本成形，线条仍然潦草，细节不稳定。

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
测试结果见 artifacts/prompt/test_metrics.json。权重不提交 Git，通过 models-v2 发布包提供（见第一节第 2 小节）。
英文烧烤场景描述能基本复制保留；口语例句 Today I went to the beach ...
曾输出 the to the beach with friends friends friends.，说明口语改写能力仍然不足。
这版可作为可运行的实验基线，不能声称比直接使用原帖更好。页面保留原帖与草稿供核对。

照片相关模型（组员开发记录，保留原文）
----------------------------------------------------------------------
新增：上传照片辅助写推文（2026-11-26）
识别改进实验：保留基线权重，独立运行 photo_tags_improved。
从本组自训基线继续训练，不加载外部预训练视觉权重：
   .\.venv\Scripts\python.exe -m scratch.photo_train --run photo_tags_improved --resume --device cuda --batch-size 32 --learning-rate 0.00015 --balanced --ema --steps 16000 --minutes 10
该命令需要先将本组基线检查点放到 artifacts/photo_tags_improved/last.pt。
--balanced 将一半采样机会分配给随机类别的正样本，另一半保持普通随机采样。
--ema 对参数和浮点统计采用 0.99 衰减的滑动平均，用于验证与推理；原始训练权重单独保存以续训。
比较/安装：.\.venv\Scripts\python.exe -m scratch.photo_compare --promote
只根据验证 macro F1 决定是否替换，原基线备份到 artifacts/photo_tags_baseline。
新版自动建议阈值只从验证集选取：至少 5 个正确建议且 precision ≥0.7，优先提高 recall。
这是有限验证集上的经验规则，不保证用户照片有相同精确率；无支持类别不自动勾选。
报告同时记录独立测试集 F1、mAP 及实际自动建议 precision/recall，见 comparison.json。
测试集只用于最终报告，不用于选模或阈值调参。支持的类别仍为 80 个，手动纠正始终保留。
改进结果：训练至 16000 步，验证集选择 13500 步，val macro F1 从 0.309848 升至 0.348583。
同一 500 张测试照片：macro F1 0.241992 → 0.268075；micro F1 0.344377 → 0.374310。
79 个有正样本类别的 mAP：0.243107 → 0.274530（无正样本类别不纳入 AP 平均）。
旧页面实际自动勾选策略 precision/recall 为 0.47125/0.24901，
新模型加验证阈值策略为 0.61085/0.17107；精确率提高但召回下降，不是各方面同时改善。
自动建议仍须逐项核对。64×64 输入和小型网络的限制仍在，尤其是小物体与少样本类别。
当前 best.pt 是改进后的权重；旧版在 artifacts/photo_tags_baseline，完整比较在
artifacts/photo_tags_improved/comparison.json。历史初轮结果保留在下文用于对照。

在帖子框下展开“从照片开始写推文”，选择静态 JPEG/PNG/WebP（最多 8 MB、2000 万像素），
点击识别，检查/修改物体标签，补充活动、人物关系等背景，然后生成三个可编辑推文版本。
支持 COCO 官方全部 80 类物体，中英文名称可筛选；不是 80 类准确率保证。
新模型是 scratch/photo_tags.py 的随机初始化 RGB CNN，独立于原六类涂鸦 CNN。
没有使用预训练 ResNet、CLIP、视觉编码器或外部图像理解 API。
标签来自官方 instances_train2017/val2017 人工标注，不从描述猜标签、不由 LLM 生成标签。
复用现有 COCO 图片及 20000/500/500 数据划分；模型输入 64×64，无法可靠辨认所有小物体。
只做多标签物体存在分类，不检测位置、数量、身份、朋友关系、烧烤/旅行等活动或地点。
人工确认是必要步骤；标签之外的信息由用户补充。Qwen 只收到确认后的文字，不会看到图片。
图片只上传到本机并在内存中处理，不写入文件、不发送到外部服务。
页面“使用此版本”才会替换帖子，之后可继续梗图推荐、改写和配图。

准备标签（已有 COCO 图片和 annotations_trainval2017.zip 时无需再下载）：
   .\.venv\Scripts\python.exe -m scratch.photo_prepare
CPU 训练：
   .\.venv\Scripts\python.exe -m scratch.photo_train --steps 5000 --minutes 10
经用户允许后可用小批量 GPU（本机此路径禁用 cuDNN，避免此前驱动问题）：
   .\.venv\Scripts\python.exe -m scratch.photo_train --resume --device cuda --batch-size 32 --steps 10000 --minutes 5
--resume 需要已有 last.pt；首次训练去掉此参数。训练/评估设备默认 CPU。
每批原子保存 artifacts/photo_tags/last.pt，按验证 macro F1 保存 best.pt。
创建 artifacts/photo_tags/STOP 可在当前批次保存后停下，续训前删除 STOP。
检查点含模型、优化器、数据指纹、CPU/可选 CUDA 随机状态，支持 CPU/GPU 间续训。
评估：.\.venv\Scripts\python.exe -m scratch.photo_train --evaluate
80 个逐类阈值仅从验证集选取；测试集独立报告 macro/micro F1 与逐类 precision/recall。
旧权重默认建议标签须达到 max(0.6, 验证阈值)，新权重使用上文验证集建议阈值，最多五个。训练正样本 <100 或验证正样本 <5 的类别
标记为样本少且不自动勾选；这些是保守交互规则，不是置信度校准或准确率保证。
模型效果仍需查看实际用户照片并评估。旧模型 Release v1 不包含这个新增照片模型。
本机初轮结果：完成 5000 步，验证集选中第 3900 步；独立测试 500 张照片，
80 类 macro F1=0.241992、micro F1=0.344377，说明整体能力仍偏弱，不等同于准确率。
测试集包含部分无正样本/极少正样本类别，逐类结果见 artifacts/photo_tags/test_metrics.json。
页面采用更保守的建议阈值，因此自动勾选结果与上述验证阈值下的 F1 不是同一配置。
固定的首张测试照片中模型建议了冲浪板，但漏掉人物，需要用户补选，不能当作成熟自动看图写作。
浏览器已验证上传、识别、标签筛选/补选、补充背景、生成三个版本与使用版本的完整流程。
Qwen 的某个候选把“海边休息”改成了“海边度假”，后者没有事实依据，说明确认标签并不能
消除语言模型的事实偏移；候选仍需人工审阅，不能作为自动发布流程。
GET /api/health 的 photo_tags 返回 80 类目录与是否存在权重。
POST /api/photo-tags 接收原始图片二进制，Content-Type 为 image/jpeg、image/png 或 image/webp。
POST /api/photo-post 接收 labels、context、style、confirmed:true，不接收图像。
训练/权重更新后重启服务清除缓存。本机推理固定 CPU，GPU 只用于显式启动的训练。

日常活动辅助写作（2026-11-26）
本机首轮结果：最终选用随机初始化的 BatchNorm CNN（1233017 参数），最佳检查点第 8200 批，最后可续训状态第 8580 批。
独立测试 1401 张：八类活动+other 九分类 accuracy=0.646681，macro F1=0.571414。
仅看 667 张属于支持八类的照片，top-1 accuracy=0.604198，八类 macro F1=0.550783。
明确建议 284/1401 张（覆盖率 20.27%），其中正确 219 张（建议精确率 77.11%）。其余保留为不确定，并非所有照片都能自动判断。
自动建议门槛按验证集选定，当前开放吃东西、跑步、骑车、鼓掌、拥抱；喝东西、坐着、使用笔记本电脑暂不自动建议，仍可手动确认。
验证集目标 80% 并未在测试集上完全达到；没有利用测试结果重新调低/调高门槛。旧 40 类模型使用不同任务与测试集，不直接比较准确率。
两个配置均只用验证集比较，选择过程保存在 selection.json；GroupNorm 对照权重/日志保留在 groupnorm_baseline/。
8 张事先按标签固定的测试照片均已通过真实二进制接口检查，所有结果保留在 artifacts/daily_activity/demo/api_results.json，包括误判和未给建议的样例。

照片 → 自训整图活动 CNN → 用户核对/修改 → 本地 Qwen 生成三个推文版本。
支持八类：吃东西、喝东西、跑步、骑自行车、鼓掌、坐着、使用笔记本电脑、拥抱。
这是整张照片的主要活动分类，不是每个人动作检测，也不能由活动判断人数、关系或心情。
旧 Stanford40 逐人物动作分类已从默认推理中停用，原权重和评估保留作为实验记录。
原人物计数/表情功能已整合到统一分析，在“识别详情与人物候选”中展示，仍需人工核对。

训练数据：https://huggingface.co/datasets/Bingsu/Human_Action_Recognition
仅下载有标注的官方 HF train（12600 张）；该仓库 test 的 labels 全为占位 0，未用于训练或评估。
从 15 个原始活动中选取上述 8 类，其余 7 类合为 other；添加原 COCO 划分中无可辨人物标注的照片作为负例。
预处理：160×160 等比缩放补灰边、保留整张照片，运行时不依赖人物框或关键点。
使用 dHash64 汉明距离≤4（含水平镜像）建相似组，每组保留分辨率最高的一张；原活动标签冲突组丢弃。
随后按原 15 类分层划分 80/10/10，COCO 负例沿用已有 train/val/test 边界并排除相似照片。
本机实际 train/val/test：10729/1401/1401 张；训练集八个支持活动各有约 663–671 张。
相似哈希去重不能保证按人物身份或原始网站来源隔离，指标仍有数据偏差风险。
另审计了统一预处理后的跨划分像素：完全相同照片为 0；跨来源近似哈希命中 9 对，经逐对查看均为不同照片（共同竖图补边/背景导致碰撞）。
数据清单保存来源 revision、原始 Parquet SHA256、样本来源、分组与划分记录。

模型为随机初始化的小型残差 CNN，不下载或加载任何外部视觉权重。
训练均衡抽取八类活动，并为 other 负例安排约 30% 的采样；使用轻微位置/缩放/翻转/色彩增强、标签平滑与 EMA。
验证 macro F1 选择 best.pt；BatchNorm 配置在验证前用固定的 512 张训练图重新计算平均权重对应的归一化统计量。
初轮还保留 GroupNorm + 全类别均衡采样配置作为验证集对照，最终只对选定配置进行独立测试。
内部为 8 类活动 + other 共 9 类。other 只是训练负例，不代表覆盖所有未知照片。
验证集先从固定候选值中选择 other 的单一 logit 偏移，补偿采样与实际负例比例的差异。
随后为每个支持活动选择建议阈值：至少 12 条建议，经验精确率≥0.8，第一/第二名分差≥0.15。
达不到条件的类别不自动建议；这不是置信度校准，也不能保证用户照片上达到 80% 正确率。
只在训练、独立测试、阈值记录与 best.pt 对应时启用自动分析，未就绪仍可手动选择活动写作。
活动必须由用户勾选确认，API 也会验证。只确认活动时可以不选择物体，给 Qwen 的活动线索不自动添加人数或人物关系；LLM 生成的草稿仍可能增加未提供的细节，必须核对。
照片仅发往本机 /api/photo-activity；Qwen 收到的是已确认的活动名称/物体/背景，不接收图片。
线索很少时 Qwen 可能只写出简短描述句；仍需用户润色或补充背景，不保证自动得到自然、无事实偏差的成稿。

准备/训练/评估：
   .\.venv\Scripts\python.exe -m pip install -r requirements-data.txt
   .\.venv\Scripts\python.exe -m scratch.activity_prepare
   .\.venv\Scripts\python.exe -m scratch.activity_train --device cpu --minutes 20 --steps 9000 --batch-size 24
   .\.venv\Scripts\python.exe -m scratch.activity_train --evaluate --device cpu
当前本机默认 CPU，先前 GPU 曾出现执行故障，本轮不自动重试 GPU。
本轮先比较两种配置，并对仍在提升的 BatchNorm 配置追加约 20 分钟 CPU 续训。
每批原子更新 artifacts/daily_activity/last.pt，保存模型、EMA、优化器、学习率状态、随机状态与训练记录。
已有权重需追加 --resume；创建 artifacts/daily_activity/STOP 可在完整批次后暂停，续训前移除 STOP。
评估输出 artifacts/daily_activity/test_metrics.json 与 calibration.json。阈值只由验证集选择，测试集不参与调参。
旧 models-v1 发布包不包含新日常活动权重。仅推理需要 best.pt、calibration.json、test_metrics.json 三个对应文件；续训另需 last.pt 与相同划分的数据。
运行网页仍使用 python server.py。

新增人物分析实验（2026-11-26）
目标：人物候选框/人数估计 → 17 个人体关键点 → 每个人的动作与可见表情候选 → 人工核对 → 推文。
与旧 80 类物体存在分类器不同，人数来自逐个框出的实例，不把 person 类分数当成人数。
PersonPoseNet：随机初始化的残差 CNN + 两层特征融合 + 中心热图、边框、17 点坐标与可见性输出。
多个人物经过峰值筛选与 IoU NMS；检测阈值只用验证集选，单次最多展示 10 个候选。
“两个人同框”不等于朋友/情侣/合影；仅输出可能合影并要求用户确认，不推断人物身份或关系。
基础举手/直立/屈膝姿态是基于预测关键点的几何规则，不伪称额外训练了这些动作标签。
动作和表情使用独立随机初始化的残差 CNN，配合 EMA 权重；不加载预训练视觉权重。
COCO：已准备 30000 训练 + 1000 验证 + 1000 测试；重新取原图，192×192 等比缩放补边。
训练样本 80% 含人物，按 1/2/3/4/5+ 人分桶采样，20% 无人物；不改变验证/测试自然分布。
使用官方非 crowd 人物框与关键点；输入尺寸下人物短边 <5 像素不参与目标，crowd 区域忽略。
因此输出的是可分辨的人物候选数，不是拥挤人群的精确计数。评估中含 crowd 的图片不计人数误差。
Stanford40：官方 40 类日常动作与人物框。按官方训练/测试划分，从训练每类留出 20% 验证；
主体人物框加 15% 上下文裁剪为 128×128，精确像素去重时测试优先。
FER2013：Hugging Face Jeneral/fer-2013 的自动转换 Parquet，仅读取字节与标签，不执行数据集脚本/pickle。
精确像素去重，测试集优先，训练集另留 2500 条验证。没有人物身份标注，不能声称按身份隔离。
表情七分类是数据集的外观类别，不是可验证的真实内心情绪。
本机实际划分（train/val/test）：动作 3200/800/5532；表情 24442/2500/7092。
首轮人物模型训练 20000 批，验证集选中第 19500 批；独立测试 1000 张照片。
IoU≥0.5：precision=0.588595，recall=0.344048，F1=0.434260。
无 crowd 测试照片的精确计数率=0.575130，人数 MAE=0.831088；
其中两人照片 93 张，精确计数率仅 0.311828，仍不能可靠识别所有两人合影。
匹配成功人物的 PCK@0.2=0.827220，仅针对已匹配的人物，不代表全图姿态成功率。
当前主要失败是漏检；总体计数率包含大量无人照片，不能代替两人/多人照片结果。
首轮动作模型验证集选中 6600 批；独立官方测试 5532 张裁剪人物图，
top-1 accuracy=0.182574，top-3 accuracy=0.357737，macro F1=0.174779。
当前动作识别很弱，只有少数高分候选可供参考，不能把“模型给出三个候选”当作可靠识别。
首轮表情模型训练 7220 批，验证集选中 7200 批；独立测试 7092 张裁剪人脸，
accuracy=0.526650，macro F1=0.480517。笑容类 F1=0.756360，其他类别仍明显混淆。
这是已裁剪 FER 人脸上的结果，不等同于完整照片中每个人的表情分析成功率。
仅在检测/关键点验证通过且双眼/鼻可见性足够、估计面部区域足够大时才尝试表情分类。
动作和表情最高分不足 0.7 或与第二名分差不足 0.2 时显示不确定；这些阈值不是置信度校准。
表情输入区域由人体关键点估计，不是可靠人脸检测器，仍可能误裁，必须人工核对。

来源与数据准备：
https://cocodataset.org/#download
https://vision.stanford.edu/Datasets/40actions.html
https://huggingface.co/datasets/Jeneral/fer-2013
HF 上也查阅了 detection-datasets/coco 与 Stanford40 镜像；当前人物/动作图片采用原始官方来源。
   .\.venv\Scripts\python.exe -m pip install -r requirements-data.txt
   .\.venv\Scripts\python.exe -m scratch.people_prepare expression
   .\.venv\Scripts\python.exe -m scratch.people_prepare action
   .\.venv\Scripts\python.exe -m scratch.people_prepare people --train 30000 --val 1000 --test 1000 --size 192
三个任务准备互不依赖，图片下载可复用本地缓存。标注文件与实际下载数、失败记录保存在各 manifest。
训练入口（串行，每批原子存盘，可续训；本机本轮 GPU 出错后默认使用 CPU）：
   .\.venv\Scripts\python.exe -m scratch.people_train expression --minutes 15 --device cpu
   .\.venv\Scripts\python.exe -m scratch.people_train action --minutes 25 --batch-size 16 --device cpu
   .\.venv\Scripts\python.exe -m scratch.person_train --minutes 50 --batch-size 8 --device cpu
以上是首次训练示例；已有权重时必须追加 --resume，程序不会覆盖现有训练进度。
同一任务追加 --resume，从 last.pt 恢复模型、优化器、CPU/CUDA 随机状态；best.pt 按验证集选择。
本机本轮动作 GPU 训练在第 2195 个已保存批次后出现 CUBLAS_STATUS_EXECUTION_FAILED，
GPU 流程已停止且未连续重试。随后改用 CPU，先续训动作 10 分钟，再训练人物至多 50 分钟：
   .\.venv\Scripts\python.exe -m scratch.people_recover
people_recover 是本轮恢复脚本，启动前需已有动作 last.pt 和完整人物数据；不是首次安装的通用入口。
原 GPU 错误日志与恢复状态在 artifacts/people_pipeline/。本轮未重试 GPU；scratch.people_run 是旧 GPU 流程，勿用于本机续训。
artifacts/people_pipeline/STOP 可暂停整个新流程；已暂停后需显式删除 STOP 再续训。
创建 artifacts/expression/STOP、artifacts/action/STOP 或 artifacts/person_pose/STOP 可暂停对应训练。
这些模块完全独立于旧 photo_tags 检查点，旧模型与四模型 Release v1 都不包含新人物分析权重。
独立测试：上面每条训练命令去掉 --minutes/--batch-size 并添加 --evaluate --device cpu。
检测报告 IoU≥0.5 的 precision/recall/F1、人数 MAE/精确计数率、匹配人物的 PCK@0.2。
此处不是 COCO 官方 AP/OKS 指标，不与官方榜单数值直接比较。
UI 中“分析人物、动作与可见表情”只发送到本机 /api/photo-people，不调用外部图像理解 API。
人物分析草稿需用户编辑/勾选确认后才会加入推文背景；Qwen 仍只接收文本。
人物模型验证 F1<0.35 时接口拒绝自动输出人数；训练检查点存在不代表效果已合格。
固定接口演示位于 artifacts/people_demo/：在模型输出前，按标注选择测试集中第一张含两人且有可见面部关键点的照片（COCO ID 91500）。
本轮正确检出两个人，但动作首选分别为阅读、使用电脑，与照片中的游戏手柄活动不符，均标为不确定；表情也均不确定。
这一张成功计数的样例不代表总体两人照片效果。analysis.json 保留实际接口输出；manual_facts_writing.json 是人工核对线索的写作检查，不能当作自动视觉输出。

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
团队复现直接下载 models-v2 发布包（见第一节第 2 小节），不需要重新准备数据或训练。
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
