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
const modes = [...document.querySelectorAll('input[name="format"]')];
const controls = [post, caption, category, seed, color, useSentiment, ...modes, byId("sample"), byId("analyze"), byId("generate")];
const sentimentLabels = {
  joy: "喜悦", sadness: "悲伤", anger: "愤怒", fear: "恐惧",
  surprise: "惊讶", affection: "关爱", neutral: "中性", other: "其他",
};
let imageUrl = null;

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
  sentiment.textContent = "推荐结果来自 Imgflip，可以打开来源或复制图片链接。";
  caption.value = "";
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
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
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
  controls.forEach((control) => { control.disabled = true; });
  say(message);
  try {
    await action();
  } catch (error) {
    say(error.message, true);
  } finally {
    controls.forEach((control) => { control.disabled = false; });
  }
}

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
    const { missing, checkpoints } = await response.json();
    const untrained = Object.entries(checkpoints).filter(([, ready]) => !ready).map(([name]) => name);
    connection.textContent = missing.length ? "缺少模型依赖" : untrained.length ? "等待模型训练" : "本地服务已连接";
    connection.dataset.state = missing.length || untrained.length ? "error" : "";
    connection.title = missing.length ? missing.join(", ") : "待训练模型 " + untrained.join(", ");
    if (!missing.length && untrained.length) {
      say("模型训练尚未完成。可取消情绪辅助后使用梗图推荐，涂鸦生成需等待训练。");
    }
  } catch {
    connection.textContent = "请启动本地服务";
    connection.dataset.state = "error";
  }
}

post.addEventListener("input", resetForPost);
byId("sample").addEventListener("click", () => {
  post.value = "熬了三个晚上，终于交了毕业论文！今天要好好庆祝一下。";
  resetForPost();
  post.focus();
});
byId("analyze").addEventListener("click", () => {
  if (!post.value.trim()) {
    say("先输入一段帖子，或点击「试用示例」。", true);
    post.focus();
    return;
  }
  run("正在匹配网络梗图并选择涂鸦类别，首次使用可能需要下载 Qwen…", async () => {
    const response = await request("/api/suggest", { post: post.value, use_sentiment: useSentiment.checked });
    const result = await response.json();
    showMemes(result.memes);
    caption.value = result.caption;
    category.value = result.category;
    clearImage();
    sentiment.textContent = result.sentiment
      ? "情绪模型判断为" + (sentimentLabels[result.sentiment.label] || result.sentiment.label)
        + "（模型分数 " + Math.round(result.sentiment.score * 100) + "%）"
      : "本次推荐未启用情绪参考。";
    say("已推荐 3 张 Imgflip 梗图，并选择涂鸦类别与配文。");
  });
});

for (const control of [category, seed, color, ...modes]) {
  control.addEventListener(control === seed ? "input" : "change", () => {
    byId("preview").classList.toggle("sticker", mode() === "sticker");
    clearImage();
    say("");
  });
}

byId("generate").addEventListener("click", () => {
  if (!seed.reportValidity()) return;
  const sticker = mode() === "sticker";
  clearImage();
  run("正在生成涂鸦，并按类别识别分数选择候选…", async () => {
    const response = await request("/api/image", {
      category: category.value, color: color.value, mode: mode(), seed: seed.value === "" ? null : seed.valueAsNumber,
    });
    imageUrl = URL.createObjectURL(await response.blob());
    art.src = imageUrl;
    art.hidden = false;
    byId("placeholder").hidden = true;
    download.disabled = false;
    const score = Math.round(Number(response.headers.get("X-Category-Score")) * 100);
    say((sticker ? "透明贴纸已生成。" : "配图已生成。") + " 种子 " + response.headers.get("X-Image-Seed")
      + "，类别识别分数 " + score + "%。");
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
  link.download = mode() === "sticker" ? "postpalette-sticker.png" : "postpalette-cover.png";
  link.href = imageUrl;
  link.click();
});

const importedPost = new URLSearchParams(location.hash.slice(1)).get("post");
if (importedPost) {
  post.value = importedPost.slice(0, 1000);
  history.replaceState(null, "", location.pathname + location.search);
}
resetForPost();
checkConnection();
