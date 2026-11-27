# PostPalette

将帖子或照片变成写作草稿、梗图推荐和配图的 Web 应用，附 Chrome 扩展入口。

**[下载最新模型权重 models-v2](https://github.com/TiantongZhang1/PostPalette/releases/tag/models-v2)** · [完整运行、数据与实验说明](README.txt)

## 组员快速运行

1. 下载本仓库最新 `main` 源码，进入包含 `server.py` 的文件夹。
2. 下载 [postpalette-models-v2.zip](https://github.com/TiantongZhang1/PostPalette/releases/download/models-v2/postpalette-models-v2.zip)。
3. 将模型 ZIP 放在源码根目录，解压后应直接得到 `artifacts/emotion/best.pt` 等路径。已有自己的模型时先备份。

```powershell
Expand-Archive -LiteralPath .\postpalette-models-v2.zip -DestinationPath . -Force
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe server.py
```

打开 <http://127.0.0.1:8765/>。本机推理使用 CPU；首次使用本地 Qwen 时需要联网下载模型。

模型包包含当前功能需要的 **9 个自训模型**：情绪 Transformer、涂鸦 CNN、条件 VAE、条件 GAN、提示词 GRU、80 类照片物体 CNN、人物/关键点网络、表情 CNN、日常活动 CNN。附校准文件、训练记录、已有测试指标和 VAE/GAN 样例，无需重新训练。

GitHub 文件列表里没有 `.pt` 是因为权重作为 **Release 附件**发布。下载源码 ZIP 或 `git clone` 不会自动包含这些附件。[SHA-256 校验文件](https://github.com/TiantongZhang1/PostPalette/releases/download/models-v2/postpalette-models-v2.zip.sha256)可用于检查下载完整性。

## 当前功能

- 帖子匹配梗图：完整热门目录、100 张内置快照、近期去重、“换一批”、来源和图片链接。
- 扩写、幽默和简洁三种风格的可编辑草稿；本地示例帖子检索。
- 上传照片统一分析物体、主要活动、人物与可见表情，人工核对后写推文。
- 自训 VAE/GAN 生成六类涂鸦贴纸，CNN 筛选候选。
- 自训 GRU 生成提示词草稿；可选外部 GPT Image API 场景配图。

外部生图需要复制 `.env.example` 为 `.env` 并自行填写 `OPENAI_API_KEY`，按 API 用量计费。其他功能无需该密钥。密钥、个人 `.env`、Qwen 缓存及训练数据均不在仓库或模型包中。

## 研究记录与限制

所有非 LLM 本地网络均从随机初始化训练。照片识别和人物计数仍有明显误差，模型输出须人工核对；表情类别不代表真实内心情绪，涂鸦分类准确率也不等于生成质量。外部 GPT Image 的课程许可尚未明确，不计作自训成果。

旧 Stanford40 动作模型与暂停的 COCO 扩散模型另放在同一 Release 的 `postpalette-experiments-v2.zip`，供查看历史实验；日常使用不需要下载。

两个包只发布 `best.pt`；继续最新批次训练还需要本地 `last.pt` 和相同数据划分。评估、训练命令和数据来源见 [README.txt](README.txt)。
