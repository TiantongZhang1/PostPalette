/* One photo workflow: analyze once, review once, then write from confirmed text. */
const photoWorkflow = (() => {
  let catalog = [];
  let photoUrl = null;
  let peopleDraft = "";
  let manualActivities = [];
  const names = { activity: "活动", objects: "物体", people: "人物" };
  const feedback = (message, error = false) => {
    byId("photo-feedback").textContent = message;
    byId("photo-feedback").dataset.state = error ? "error" : "";
  };
  const selectedLabels = () => [...byId("photo-tags").querySelectorAll("input:checked")].map((input) => input.value);

  function syncReview() {
    const labels = selectedLabels();
    const activity = byId("activity-choice");
    const context = byId("photo-context").value.trim();
    const parts = [];
    if (activity.value) parts.push(activity.selectedOptions[0].textContent);
    if (labels.length) parts.push(labels.length + " 个物体");
    if (context) parts.push("补充说明");
    const valid = parts.length > 0 && labels.length <= 20;
    byId("photo-review-summary").textContent = labels.length > 20 ? "物体最多选择 20 个，请取消多余标签。"
      : parts.length ? "将用于写作：" + parts.join(" · ") : "选择活动、物体，或填写补充说明。";
    byId("photo-write").disabled = appBusy || !valid || !byId("photo-confirm").checked;
    byId("photo-tags-empty").hidden = labels.length > 0;
    byId("use-people").disabled = appBusy || !peopleDraft || byId("photo-context").value.includes(peopleDraft);
  }

  function invalidate() {
    byId("photo-confirm").checked = false;
    byId("photo-drafts").replaceChildren();
    syncReview();
  }

  function filterLabels() {
    const query = byId("photo-filter").value.trim().toLowerCase();
    for (const label of byId("photo-other-tags").querySelectorAll("label")) {
      label.hidden = !label.textContent.toLowerCase().includes(query) && !label.querySelector("input").value.includes(query);
    }
  }

  function positionLabel(label) {
    const input = label.querySelector("input");
    label.hidden = false;
    byId(input.checked ? "photo-selected-tags" : "photo-other-tags").append(label);
  }

  function showLabels(labels) {
    byId("photo-selected-tags").replaceChildren();
    byId("photo-other-tags").replaceChildren();
    for (const tag of labels) {
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "checkbox"; input.value = tag.id; input.checked = Boolean(tag.suggested);
      label.append(input, tag.name);
      if (tag.score !== undefined) label.title = "模型分数 " + Math.round(tag.score * 100) + "，不是正确概率";
      positionLabel(label);
    }
    filterLabels(); syncReview();
  }

  function activityNote(result) {
    byId("activity-choice").value = result?.accepted ? result.suggestion : "";
    byId("activity-note").textContent = result?.accepted
      ? "已填入活动候选，请核对或修改。"
      : "活动暂不确定；可手动选择，也可直接填写补充说明。";
    if (manualActivities.length && !result?.accepted) {
      byId("activity-note").textContent += " " + manualActivities.join("、") + "目前需手动选择。";
    }
    byId("activity-details").textContent = result
      ? result.candidates.map((c) => c.name + " " + Math.round(c.score * 100) + "分").join("、") + "。分数不是正确概率。"
      : "活动分析暂不可用。";
  }

  function showPeople(result) {
    peopleDraft = "";
    byId("people-results").replaceChildren();
    byId("people-overlay").hidden = true;
    byId("use-people").hidden = true;
    byId("people-status").textContent = result
      ? result.summary + " 人物候选默认不用于写作，需检查后采用。"
      : "人物分析暂不可用，可直接在补充说明中填写人数或动作。";
    if (!result) return;
    const lines = [result.summary];
    const image = byId("photo-preview");
    for (const person of result.people) {
      const clues = [];
      if (person.pose_hint) clues.push(person.pose_hint);
      if (person.expression && !person.expression.uncertain) clues.push("可见表情候选：" + person.expression.candidates[0].name);
      const text = "人物 " + person.number + "：" + (clues.length ? clues.join("；") : "没有可靠的姿态或表情线索");
      const paragraph = document.createElement("p"); paragraph.textContent = text;
      byId("people-results").append(paragraph);
      if (clues.length) lines.push(text);
      if (person.expression) {
        const candidates = document.createElement("p"); candidates.className = "hint";
        candidates.textContent = "表情候选：" + person.expression.candidates.map((c) => c.name + " " + Math.round(c.score*100) + "分").join("、")
          + (person.expression.uncertain ? "（不确定，不加入人物描述）" : "（仍需核对）");
        byId("people-results").append(candidates);
      }
      if (person.expression_region && image.complete && image.naturalWidth) {
        const face = document.createElement("canvas"); face.width = 80; face.height = 80;
        face.setAttribute("aria-label", "估计面部区域；裁剪错误时请忽略表情候选");
        const [x1,y1,x2,y2] = person.expression_region;
        face.getContext("2d").drawImage(image,x1*image.naturalWidth,y1*image.naturalHeight,(x2-x1)*image.naturalWidth,(y2-y1)*image.naturalHeight,0,0,80,80);
        byId("people-results").append(face);
      }
    }
    if (result.count_estimate > 0) {
      peopleDraft = lines.join("\n");
      byId("use-people").hidden = false;
    }
    if (image.complete && image.naturalWidth) {
      const canvas = byId("people-overlay");
      const scale = Math.min(640 / image.naturalWidth, 640 / image.naturalHeight);
      canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
      canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
      const context = canvas.getContext("2d"); context.drawImage(image,0,0,canvas.width,canvas.height);
      context.strokeStyle = "#5936b8"; context.fillStyle = "#5936b8"; context.lineWidth = 3; context.font = "20px sans-serif";
      for (const person of result.people) {
        const [x1,y1,x2,y2] = person.box;
        context.strokeRect(x1*canvas.width,y1*canvas.height,(x2-x1)*canvas.width,(y2-y1)*canvas.height);
        context.fillText(String(person.number),x1*canvas.width+3,Math.max(22,y1*canvas.height+20));
      }
      canvas.hidden = false;
    }
  }

  async function runPhoto(message, action) {
    feedback(message);
    await run(message, async () => {
      try { await action(); }
      catch (error) { feedback(error.message, true); throw error; }
    });
  }

  byId("photo-tags").addEventListener("change", (event) => {
    if (!event.target.matches('input[type="checkbox"]')) return;
    positionLabel(event.target.closest("label")); filterLabels(); invalidate();
  });
  byId("photo-filter").addEventListener("input", filterLabels);
  byId("activity-choice").addEventListener("change", invalidate);
  byId("photo-context").addEventListener("input", invalidate);
  byId("photo-style").addEventListener("change", () => { byId("photo-drafts").replaceChildren(); syncReview(); });
  byId("photo-confirm").addEventListener("change", syncReview);

  byId("photo-file").addEventListener("change", () => {
    if (photoUrl) URL.revokeObjectURL(photoUrl);
    photoUrl = null;
    byId("photo-preview").removeAttribute("src"); byId("photo-preview").hidden = true;
    byId("photo-context").value = ""; byId("activity-choice").value = ""; byId("photo-filter").value = "";
    byId("photo-more-tags").open = false; byId("photo-analysis-details").open = false;
    byId("photo-analyze").textContent = "分析照片";
    showLabels(catalog); showPeople(null); activityNote(null);
    byId("photo-analysis-status").replaceChildren(); invalidate();
    const file = byId("photo-file").files[0];
    if (!file) { feedback("可以添加照片，也可以手动补充写作线索。"); return; }
    if (!["image/jpeg","image/png","image/webp"].includes(file.type) || file.size > 8*1024*1024) {
      byId("photo-file").value = ""; feedback("请选择不超过 8 MB 的 JPEG、PNG 或 WebP 照片。", true); return;
    }
    photoUrl = URL.createObjectURL(file); byId("photo-preview").src = photoUrl; byId("photo-preview").hidden = false;
    feedback("照片已添加，点击「分析照片」提取线索。");
  });

  byId("photo-analyze").addEventListener("click", () => {
    const file = byId("photo-file").files[0];
    if (!file) { feedback("请先添加一张照片。", true); return; }
    const previousActivity = byId("activity-choice").value;
    const previousLabels = new Set(selectedLabels());
    invalidate();
    runPhoto("正在本机统一分析活动、物体和人物…", async () => {
      const response = await request("/api/photo-analyze",file); const result = await response.json();
      activityNote(result.activity);
      if (!result.activity && previousActivity) {
        byId("activity-choice").value = previousActivity;
        byId("activity-note").textContent = "活动分析暂不可用，已保留你手动选择的活动，请核对。";
      }
      showLabels(result.objects?.predictions || catalog.map((tag) => ({...tag,suggested:previousLabels.has(tag.id)})));
      showPeople(result.people);
      byId("photo-analysis-status").replaceChildren();
      for (const key of Object.keys(names)) {
        const item = document.createElement("li"); const error = result.errors.find((e) => e.module === key);
        item.textContent = names[key] + "：" + (error ? error.message : "已完成分析，结果需核对");
        byId("photo-analysis-status").append(item);
      }
      byId("photo-analyze").textContent = "重新分析照片"; invalidate();
      const completed = Object.keys(names).filter((key) => result[key]).length;
      const message = completed === 3 ? "分析完成。核对下方线索后，只需统一确认一次。"
        : completed ? "已保留可用线索，部分分析暂不可用；可在详情查看原因或手动补充。"
        : "自动分析暂不可用，可以手动补充线索后写作。";
      feedback(message, !completed); say(message, !completed);
    });
  });

  byId("use-people").addEventListener("click", () => {
    if (!peopleDraft || appBusy) return;
    const value = [byId("photo-context").value.trim(),peopleDraft].filter(Boolean).join("\n");
    if (value.length > 500) { feedback("加入人物描述后会超过 500 字，请先精简补充说明。", true); return; }
    byId("photo-context").value = value; invalidate();
    feedback("人物候选已加入补充说明，请核对、修改后统一确认。");
  });

  byId("photo-write").addEventListener("click", () => {
    if (!byId("photo-confirm").checked) { feedback("请先核对并确认写作线索。", true); return; }
    const labels = selectedLabels(); const activity = byId("activity-choice").value || null; const context = byId("photo-context").value.trim();
    if ((!labels.length && !activity && !context) || labels.length > 20) { feedback("至少补充一条线索，物体最多选择 20 个。", true); return; }
    runPhoto("正在根据已确认线索生成三个草稿；首次加载可能稍慢…", async () => {
      const response = await request("/api/photo-post", {labels,activity,context,style:byId("photo-style").value,confirmed:true,activity_confirmed:Boolean(activity)});
      showVersions(await response.json(),"photo-drafts");
      feedback("三个草稿已生成，可编辑、复制，或使用此版本继续推荐梗图与生成配图。");
      byId("photo-drafts").scrollIntoView({behavior:"smooth",block:"start"});
    });
  });

  syncReview();
  return {syncReview, configure(photoTags, dailyActivity) {
    catalog = photoTags.labels; manualActivities = (dailyActivity?.manual_only || []).map((tag) => tag.name);
    if (!byId("photo-tags").querySelector('input[type="checkbox"]')) showLabels(catalog);
    byId("photo-model-status").textContent = "活动、物体、人物一次分析，线索统一核对后写作。照片只在本机处理。";
  }};
})();
