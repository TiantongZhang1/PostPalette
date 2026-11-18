const workspace = "http://127.0.0.1:8765/";

chrome.action.onClicked.addListener(async (tab) => {
  let text = "";
  if (tab.id !== undefined) {
    try {
      const [selection] = await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: () => window.getSelection()?.toString().trim().slice(0, 1000) ?? "",
      });
      text = selection?.result ?? "";
    } catch {
      // Restricted pages still open the workspace for manual input.
    }
  }
  const fragment = text ? "#" + new URLSearchParams({ post: text }) : "";
  await chrome.tabs.create({ url: workspace + fragment });
});
