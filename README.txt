PostPalette · 本地帖子配图工作台

Windows 启动（Python 3.12，PowerShell）
1. 在 PostPalette 文件夹打开 PowerShell。
2. 建立独立环境：
   python -m venv .venv
3. NVIDIA RTX 5080 等支持 CUDA 12.8 的显卡，先安装 CUDA 版 PyTorch：
   .\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
4. 安装应用依赖：
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
5. 启动：
   .\.venv\Scripts\python.exe server.py
6. 用 Chrome 或 Edge 打开 http://127.0.0.1:8765/

使用 CPU 的机器可以跳过第 3 步，或先从 PyTorch 的 CPU 索引安装 torch。
CPU 出图明显更慢，建议在演示机器上优先验证显卡。
首次使用需要访问 Hugging Face 和 GitHub 下载模型，缓存位于 .model-cache。
建议留至少 25 GB 磁盘空间，并分别运行一次文字、配图、透明贴纸流程完成预热。
后续运行直接执行第 5、6 步。Ctrl+C 关闭服务。

Chrome 扩展
1. 保持本地服务运行在默认端口 8765。
2. 在 chrome://extensions/ 开启开发者模式，点击“加载已解压的扩展程序”。
3. 选择 PostPalette/extension 文件夹。可在 Chrome 工具栏固定 PostPalette。
4. 在普通网页选中帖子文字，再点扩展图标。工作台会带入选区文字。
   也可以直接在工作台粘贴文字。浏览器设置页等特殊页面不给扩展读取选区。

生成与比较
1. 输入帖子或点“试用示例”，保持“使用情绪模型辅助推荐”勾选。
2. 点“推荐梗图”。结果包含 Imgflip 预览、原始模板页面和图片链接。右侧可编辑绘图提示词。
3. 选配图或透明贴纸、选择风格、点“生成图片”。下载按钮输出 PNG。
4. 文本对照：对同一帖子取消情绪辅助，再生成一次，对比梗图排名与贴切程度。
5. 图像对照：固定同一风格和种子，例如 2026，分别使用直接描述和模型改写后的
   提示词生成配图。对同一贴纸提示词和种子输出 PNG，检查背景与主体边缘。

课程主题与模型
- Transformer Encoder：cardiffnlp/twitter-xlm-roberta-base-sentiment 给出
  正向、中性或负向判定。中文帖先由 Qwen 翻译成英文，再分析情绪和生成建议。
- LLM：Qwen/Qwen2.5-1.5B-Instruct 根据帖子和情绪信号重排候选梗图，并写英文绘图提示词。
- Diffusion：stabilityai/sdxl-turbo 生成 512×512 图像。
- CNN：rembg 的 U²-Net ONNX 模型为贴纸提取 alpha 通道。

本地服务仅监听回环地址，项目不需要 API 密钥。首次联网用于下载依赖和权重。
梗图候选来自 Imgflip 热门模板目录（https://imgflip.com/api）。应用对带有用途标注的模板
做关键词初选，用当前热度打破平局，再让 Qwen 排序。目录暂不可用时用内置模板元数据；
预览仍需要联网，界面保留来源链接。向 Imgflip 请求目录和缩略图时不发送帖子文字。
模型首次加载可能较慢。若生成失败，页面会给出失败环节，终端会记录诊断信息。
分享源码时可排除 .venv、.model-cache 和 __pycache__，接收方按以上步骤安装。
