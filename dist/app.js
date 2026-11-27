const byId = (id) => document.getElementById(id);
const post = byId("post");
const caption = byId("caption");
const category = byId("category");
const seed = byId("seed");
const color = byId("color");
const status = byId("status");
const art = byId("art");
const download = byId("download");
const sentiment = byId("sentiment");
const useSentiment = byId("use-sentiment");
const connection = byId("connection");
const generator = byId("generator");
const doodleModel = byId("doodle-model");
const scenePrompt = byId("scene-prompt");
const modes = [...document.querySelectorAll('input[name="format"]')];
const controls = [post, caption, category, seed, color, generator, doodleModel, scenePrompt, useSentiment, ...modes, byId("sample"), byId("analyze"), byId("refresh-memes"), byId("generate"), byId("build-prompt"), byId("rewrite-style"), byId("rewrite-post"), byId("find-references"), byId("restore-post"), byId("photo-file"), byId("photo-context"), byId("photo-style"), byId("photo-analyze"), byId("photo-write"), byId("photo-confirm"), byId("activity-choice"), byId("photo-filter"), byId("use-people")];
const doodleModelNames = { vae: "条件 VAE", gan: "条件 GAN" };
const sentimentLabels = {
  joy: "喜悦", sadness: "悲伤", anger: "愤怒", fear: "恐惧",
  surprise: "惊讶", affection: "关爱", neutral: "中性", other: "其他",
};
let imageUrl = null;
let previousPost = null;
let appBusy = false;
let seenMemeIds = [];
let displayedMemeIds = [];
let memeTotal = 0;
let memeRemaining = 0;
let memePost = "";
let recentMemeIds = [];
try {
  const saved = JSON.parse(localStorage.getItem("postpalette-recent-memes") || "[]");
  if (Array.isArray(saved)) recentMemeIds = [...new Set(saved.filter((id) => typeof id === "string" && /^[0-9]{1,20}$/.test(id)))].slice(-18);
} catch { /* Recommendations also work when browser storage is unavailable. */ }

function syncMemeControls() {
  byId("refresh-memes").disabled = appBusy || !displayedMemeIds.length
    || memePost !== post.value.trim() || memeRemaining < 3 || seenMemeIds.length > 117;
}

function say(message, error = false) {
  status.textContent = message;
  status.dataset.state = error ? "error" : "";
}

function clearImage() {
  if (imageUrl) URL.revokeObjectURL(imageUrl);
  imageUrl = null;
  art.removeAttribute("src");
  art.hidden = true;
  byId("placeholder").hidden = false;
  download.disabled = true;
}

function resetForPost() {
  byId("count").textContent = post.value.length + " / 1000";
  byId("memes").replaceChildren();
  seenMemeIds = [];
  displayedMemeIds = [];
  memePost = "";
  byId("meme-info").textContent = "从完整热门目录匹配梗图；自动避开最近看过的模板。";
  syncMemeControls();
  sentiment.textContent = "推荐结果来自 Imgflip，可以打开来源或复制图片链接。";
  caption.value = "";
  scenePrompt.value = "";
  byId("prompt-info").textContent = "先填写左侧帖子，再生成并检查提示词。";
  byId("prompt-details").hidden = true;
  byId("prompt-draft").textContent = "";
  byId("rewrite-results").replaceChildren();
  byId("reference-results").replaceChildren();
  byId("reference-info").textContent = "参考库为 24 条 AI 编写的项目示例，非真实用户帖子或实时网络搜索；按词面相似度检索。";
  clearImage();
  say("");
}

async function request(path, data) {
  if (location.protocol === "file:") {
    throw new Error("请运行 python server.py，再打开 http://127.0.0.1:8765/。");
  }
  let response;
  try {
    response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": data instanceof Blob ? data.type : "application/json" },
      body: data instanceof Blob ? data : JSON.stringify(data),
    });
  } catch {
    throw new Error("与本地服务的连接中断，请确认启动终端仍在运行。");
  }
  if (!response.ok) {
    const result = await response.json().catch(() => ({}));
    throw new Error(result.error || "请求失败，请查看启动终端。");
  }
  return response;
}

async function run(message, action) {
  appBusy = true;
  controls.forEach((control) => { control.disabled = true; });
  byId("photo-tags").querySelectorAll("input").forEach((input) => { input.disabled = true; });
  say(message);
  try {
    await action();
  } catch (error) {
    say(error.message, true);
  } finally {
    appBusy = false;
    controls.forEach((control) => { control.disabled = false; });
    byId("photo-tags").querySelectorAll("input").forEach((input) => { input.disabled = false; });
    photoWorkflow.syncReview();
    syncMemeControls();
  }
}

function requirePost() {
  if (post.value.trim()) return true;
  say("先输入一段帖子，或点击「试用示例」。", true);
  post.focus();
  return false;
}

function textAction(label, action) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "link";
  button.textContent = label;
  button.addEventListener("click", action);
  return button;
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    say("已复制文字。");
  } catch {
    say("浏览器未允许复制，请选中文字后手动复制。", true);
  }
}

function showVersions(result, containerId = "rewrite-results") {
    byId(containerId).replaceChildren();
    result.versions.forEach((version, index) => {
      const card = document.createElement("article");
      card.className = "text-result";
      const heading = document.createElement("h4");
      heading.textContent = "版本 " + (index + 1);
      const editor = document.createElement("textarea");
      editor.maxLength = 1000;
      editor.value = version;
      editor.setAttribute("aria-label", "编辑改写版本 " + (index + 1));
      const actions = document.createElement("div");
      actions.className = "text-actions";
      actions.append(textAction("使用此版本", () => {
        if (post.disabled) { say("请等待当前操作完成。", true); return; }
        if (!editor.value.trim()) { say("改写内容不能为空。", true); return; }
        previousPost = post.value;
        post.value = editor.value;
        byId("restore-post").hidden = false;
        resetForPost();
        post.focus();
        say("已替换帖子。可以继续推荐梗图或生成配图提示词，也可以恢复替换前原文。");
      }), textAction("复制", () => copyText(editor.value)));
      card.append(heading, editor, actions);
      byId(containerId).append(card);
    });
    say(result.notice);
}

byId("rewrite-post").addEventListener("click", () => {
  if (!requirePost()) return;
  run("正在本机生成三个改写版本，CPU 首次加载可能需要一些时间…", async () => {
    const response = await request("/api/rewrite", { post: post.value, style: byId("rewrite-style").value });
    showVersions(await response.json());
  });
});

byId("restore-post").addEventListener("click", () => {
  if (previousPost === null) return;
  post.value = previousPost;
  previousPost = null;
  byId("restore-post").hidden = true;
  resetForPost();
  say("已恢复替换前原文。");
});

byId("find-references").addEventListener("click", () => {
  if (!requirePost()) return;
  run("正在检索本地参考示例…", async () => {
    const response = await request("/api/references", { post: post.value });
    const result = await response.json();
    byId("reference-results").replaceChildren();
    byId("reference-info").textContent = result.corpus_description + " 当前共 " + result.corpus_count + " 条；分数是词面匹配分数，不是正确概率。";
    for (const match of result.matches) {
      const card = document.createElement("article");
      card.className = "text-result";
      const heading = document.createElement("h4");
      heading.textContent = match.topic + " · 匹配分数 " + match.score.toFixed(3);
      const text = document.createElement("p");
      text.textContent = match.text;
      const source = document.createElement("p");
      source.className = "hint";
      source.textContent = match.source + "；匹配词：" + match.matched_terms.join("、");
      card.append(heading, text, source, textAction("复制参考文字", () => copyText(match.text)));
      byId("reference-results").append(card);
    }
    say(result.matches.length ? "已找到 " + result.matches.length + " 条本地参考，仅供表达参考。" : "本地示例库中没有足够相关的内容，可以换关键词；当前不进行网络搜索。");
  });
});

function showMemes(memes) {
  byId("memes").replaceChildren(...memes.map((meme) => {
    const card = document.createElement("article");
    card.className = "meme";
    const link = document.createElement("a");
    link.href = meme.source_url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    const image = document.createElement("img");
    image.alt = "";
    image.src = meme.image_url;
    image.loading = "lazy";
    image.addEventListener("error", () => { image.hidden = true; });
    link.append(image, meme.name);
    const button = document.createElement("button");
    button.className = "link";
    button.type = "button";
    button.textContent = "复制图片链接";
    button.setAttribute("aria-label", "复制 " + meme.name + " 的图片链接");
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(meme.image_url);
        say("已复制 " + meme.name + " 的图片链接。");
      } catch {
        say("浏览器未允许复制，可以点梗图标题打开来源。", true);
      }
    });
    card.append(link, button);
    return card;
  }));
}

function mode() {
  return modes.find((input) => input.checked).value;
}

async function checkConnection() {
  try {
    if (location.protocol === "file:") throw new Error();
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error();
    const { missing, checkpoints, image_api: imageApi, photo_tags: photoTags, people_models: peopleModels, daily_activity: dailyActivity } = await response.json();
    photoWorkflow.configure(photoTags, dailyActivity, peopleModels);
    byId("api-status").textContent = imageApi.configured
      ? "图像 API 已配置 · " + imageApi.model + " · " + imageApi.quality
      : "图像 API 尚未配置：请在项目 .env 中填写 OPENAI_API_KEY 后重启服务。";
    const untrained = Object.entries(checkpoints).filter(([, ready]) => !ready).map(([name]) => name);
    connection.textContent = missing.length ? "缺少模型依赖" : untrained.length ? "部分模型待训练" : "本地服务已连接";
    connection.dataset.state = missing.length || untrained.length ? "error" : "";
    connection.title = missing.length ? missing.join(", ") : "待训练模型 " + untrained.join(", ");
    if (!missing.length && untrained.length) {
      say("尚无权重的模型：" + untrained.join("、") + "。其他已训练功能仍可使用。");
    }
  } catch {
    connection.textContent = "请启动本地服务";
    connection.dataset.state = "error";
  }
}

post.addEventListener("input", resetForPost);
byId("sample").addEventListener("click", () => {
    post.value = "今天和三个朋友去公园烧烤了，大家围着烤架聊天，差点把肉烤糊。";
  resetForPost();
  post.focus();
});
function recommendMemes(refresh = false) {
  if (!requirePost()) return;
  if (refresh && memePost !== post.value.trim()) return;
  // Only template IDs are stored, never post content. Each browser has its own history.
  const exclusions = refresh ? seenMemeIds : recentMemeIds;
  run(refresh ? "正在寻找本轮未展示过的梗图…" : "正在匹配完整梗图库，首次使用可能需要加载 Qwen…", async () => {
    const response = await request("/api/suggest", { post: post.value, use_sentiment: useSentiment.checked, exclude_ids: exclusions });
    const result = await response.json();
    showMemes(result.memes);
    displayedMemeIds = result.memes.map((meme) => meme.id);
    seenMemeIds = [...new Set([...exclusions, ...displayedMemeIds])];
    recentMemeIds = [...new Set([...recentMemeIds.filter((id) => !displayedMemeIds.includes(id)), ...displayedMemeIds])].slice(-18);
    try { localStorage.setItem("postpalette-recent-memes", JSON.stringify(recentMemeIds)); } catch { /* optional */ }
    memePost = post.value.trim();
    memeTotal = result.catalog.total;
    memeRemaining = result.catalog.remaining;
    const source = { live: "在线目录", cached: "上次成功缓存，在线更新暂不可用", bundled: "项目内置目录，在线更新暂不可用", fallback: "基础备用目录" }[result.catalog.mode] || "模板目录";
    byId("meme-info").textContent = "当前图库 " + memeTotal + " 张 · " + source + "。"
      + (memeRemaining < 3 || seenMemeIds.length > 117 ? " 本轮浏览已结束，可点击「推荐梗图」开始新一轮。" : " 换一批会避开本轮已排除和展示的模板。");
    if (!refresh) {
      caption.value = result.caption;
      category.value = result.category;
      clearImage();
    }
    sentiment.textContent = result.sentiment
      ? "情绪模型判断为" + (sentimentLabels[result.sentiment.label] || result.sentiment.label)
        + "（模型分数 " + Math.round(result.sentiment.score * 100) + "%）"
      : "本次推荐未启用情绪参考。";
    say(refresh ? "已换成 3 张未展示的梗图。" : "已从扩展图库推荐 3 张梗图，并选择涂鸦类别与配文。");
  });
}
byId("analyze").addEventListener("click", () => recommendMemes());
byId("refresh-memes").addEventListener("click", () => recommendMemes(true));

for (const control of [category, seed, color, doodleModel, ...modes]) {
  control.addEventListener(control === seed ? "input" : "change", () => {
    byId("preview").classList.toggle("sticker", mode() === "sticker");
    clearImage();
    say("");
  });
}

generator.addEventListener("change", () => {
  byId("scene-options").hidden = generator.value !== "scene";
  byId("doodle-options").hidden = generator.value === "scene";
  byId("seed-options").hidden = generator.value === "scene";
  byId("preview").classList.toggle("sticker", generator.value !== "scene" && mode() === "sticker");
  clearImage();
  say("");
});
scenePrompt.addEventListener("input", clearImage);

byId("build-prompt").addEventListener("click", () => {
  if (!post.value.trim()) {
    say("先输入一段帖子。", true);
    post.focus();
    return;
  }
  run("正在用自训模型生成提示词；中文帖子会先在本机翻译，首次加载可能较慢…", async () => {
    const response = await request("/api/prompt", { post: post.value });
    const result = await response.json();
    scenePrompt.value = result.prompt;
    byId("prompt-draft").textContent = result.draft || "模型未生成有效草稿，请编辑提示词。";
    byId("prompt-details").hidden = false;
    byId("prompt-info").textContent = "自训模型第 " + result.step + " 步"
      + (result.translated ? " · 已通过本地 Qwen 翻译" : "") + "。" + result.warning;
    clearImage();
    say("提示词已生成。检查描述后点击「生成图片」。");
  });
});

byId("generate").addEventListener("click", () => {
  if (generator.value === "scene") {
    const description = scenePrompt.value.trim();
    if (!description) {
      say("请先点击「根据帖子生成提示词」，或输入自己的画面描述。", true);
      scenePrompt.focus();
      return;
    }
    clearImage();
    run("正在调用 GPT 图像 API，通常需要一些时间，请勿重复提交…", async () => {
      const response = await request("/api/gpt-image", { prompt: description });
      imageUrl = URL.createObjectURL(await response.blob());
      art.src = imageUrl;
      art.hidden = false;
      byId("placeholder").hidden = true;
      download.disabled = false;
      say("场景配图已生成 · " + response.headers.get("X-Image-Model") + " · 1024×1024。");
    });
    return;
  }
  if (!seed.reportValidity()) return;
  const sticker = mode() === "sticker";
  clearImage();
  run("正在生成涂鸦，并按类别识别分数选择候选…", async () => {
    const response = await request("/api/image", {
      category: category.value, color: color.value, mode: mode(), seed: seed.value === "" ? null : seed.valueAsNumber,
      generator: doodleModel.value,
    });
    imageUrl = URL.createObjectURL(await response.blob());
    art.src = imageUrl;
    art.hidden = false;
    byId("placeholder").hidden = true;
    download.disabled = false;
    const score = Math.round(Number(response.headers.get("X-Category-Score")) * 100);
    say((sticker ? "透明贴纸已生成。" : "配图已生成。") + " " + (doodleModelNames[doodleModel.value] || doodleModel.value)
      + "，种子 " + response.headers.get("X-Image-Seed") + "，类别识别分数 " + score + "%。");
  });
});

byId("copy-caption").addEventListener("click", async () => {
  if (!caption.value.trim()) {
    say("先生成或输入一段配文。", true);
    return;
  }
  try {
    await navigator.clipboard.writeText(caption.value);
    say("已复制配文。");
  } catch {
    caption.focus();
    caption.select();
    say("已选中配文，按 Ctrl+C 或 ⌘C 复制。");
  }
});

download.addEventListener("click", () => {
  if (!imageUrl) return;
  const link = document.createElement("a");
  link.download = generator.value === "scene" ? "postpalette-scene.png"
    : mode() === "sticker" ? "postpalette-sticker.png" : "postpalette-cover.png";
  link.href = imageUrl;
  link.click();
});

const importedPost = new URLSearchParams(location.hash.slice(1)).get("post");
if (importedPost) {
  post.value = importedPost.slice(0, 1000);
  history.replaceState(null, "", location.pathname + location.search);
}
resetForPost();
document.addEventListener("DOMContentLoaded", checkConnection);
