// The chat portal: a conversation with the agent over server-sent events, with an optional voice conversation
// (voice.js) on top. No framework and no build step: the server serves this folder as it is.

const $ = (id) => document.getElementById(id);
const log = $("log");
const scroller = $("scroll");
const input = $("input");
const sendButton = $("send");
const talk = $("talk");
const caption = $("caption");
const fineprint = $("fineprint");
const dock = document.querySelector(".dock");

const store = {
  get(key) {
    try {
      return localStorage.getItem(key) || "";
    } catch {
      return "";
    }
  },
  set(key, value) {
    try {
      if (value) localStorage.setItem(key, value);
      else localStorage.removeItem(key);
    } catch {
      /* private mode: forget on reload */
    }
  },
};

function newThreadId() {
  const random = crypto.randomUUID ? crypto.randomUUID() : `${Math.random().toString(36).slice(2)}${Date.now().toString(36)}`;
  return random.replace(/[^A-Za-z0-9]/g, "").slice(0, 40);
}

const state = {
  config: null,
  thread: store.get("agent.thread") || newThreadId(),
  code: store.get("agent.code"),
  busy: false,
  approval: null, // the approval waiting for an answer: {calls: [{id, name, args, summary}]}
  approvalCard: null,
  voice: null,
  lastBubble: null,
};

// ------------------------------------------------------------------ the API

function headers(extra = {}) {
  return state.code ? { ...extra, "X-Access-Code": state.code } : extra;
}

async function failure(res) {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
  } catch {
    /* not JSON */
  }
  return `${res.status} ${res.statusText}`;
}

/** fetch under /api with the access code, asking for the code again when it is refused. */
async function api(path, init = {}) {
  const res = await fetch(`/api${path}`, { ...init, headers: headers(init.headers) });
  if (res.status === 401) {
    await askForCode(true);
    return api(path, init);
  }
  return res;
}

async function getJSON(path) {
  const res = await api(path);
  if (!res.ok) throw new Error(await failure(res));
  return res.json();
}

/** POSTs and hands each server-sent event to onEvent until the stream ends. */
async function stream(path, body, onEvent) {
  const res = await api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok || !res.body) throw new Error(await failure(res));
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const data = buffer
        .slice(0, cut)
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      buffer = buffer.slice(cut + 2);
      if (data) onEvent(JSON.parse(data));
    }
    if (done) break;
  }
}

function askForCode(refused) {
  return new Promise((resolve) => {
    const gate = $("gate");
    $("gate-error").textContent = refused && state.code ? "That code did not work. Check it and try again." : "";
    $("gate-code").value = "";
    gate.oncancel = (event) => event.preventDefault(); // the portal is no use without it
    $("gate-form").onsubmit = () => {
      state.code = $("gate-code").value.trim();
      store.set("agent.code", state.code);
      resolve();
    };
    if (!gate.open) gate.showModal();
  });
}

// ------------------------------------------------------------------ rendering

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function scrollDown() {
  scroller.scrollTop = scroller.scrollHeight;
}

function add(node) {
  log.querySelector(".hello")?.remove();
  log.append(node);
  scrollDown();
  return node;
}

const escapeHtml = (text) => text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

function inline(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\s][^*]*?)\*(?!\w)/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
}

/** Enough Markdown for chat replies: paragraphs, lists, bold, italics, code and links. Everything is escaped first. */
function markdown(text) {
  const out = [];
  text.split("```").forEach((part, i) => {
    if (i % 2) {
      out.push(`<pre><code>${escapeHtml(part.replace(/^\w*\n/, ""))}</code></pre>`);
      return;
    }
    let list = null;
    for (const raw of part.split("\n")) {
      const line = raw.trimEnd();
      const bullet = line.match(/^\s*[-*•]\s+(.*)/);
      const numbered = line.match(/^\s*\d+[.)]\s+(.*)/);
      const kind = bullet ? "ul" : numbered ? "ol" : null;
      if (kind) {
        if (list !== kind) {
          if (list) out.push(`</${list}>`);
          out.push(`<${kind}>`);
          list = kind;
        }
        out.push(`<li>${inline((bullet || numbered)[1])}</li>`);
        continue;
      }
      if (list) {
        out.push(`</${list}>`);
        list = null;
      }
      if (!line.trim()) continue;
      const heading = line.match(/^#{1,6}\s+(.*)/);
      out.push(heading ? `<p><strong>${inline(heading[1])}</strong></p>` : `<p>${inline(line)}</p>`);
    }
    if (list) out.push(`</${list}>`);
  });
  return out.join("");
}

function addUser(text) {
  add(el("div", "msg user", text));
}

function addAgent(text) {
  const bubble = add(el("div", "msg agent"));
  bubble.innerHTML = markdown(text);
  state.lastBubble = bubble;
  return bubble;
}

function notice(text) {
  add(el("div", "notice", text));
}

/** Three dots while the agent works with nothing to show yet: before its first words, and between a tool and its reply. */
let thinkingDots = null;

function showThinking() {
  if (thinkingDots) return;
  thinkingDots = el("div", "thinking");
  thinkingDots.setAttribute("aria-label", "The agent is working");
  thinkingDots.append(el("i"), el("i"), el("i"));
  add(thinkingDots);
}

function hideThinking() {
  thinkingDots?.remove();
  thinkingDots = null;
}

/** The agent's words in one run. Text after a tool call starts a new bubble, so the order on screen is the order
 * things happened. */
class Reply {
  bubble = null;
  text = "";

  append(chunk) {
    if (!this.bubble) {
      this.bubble = add(el("div", "msg agent streaming"));
      this.text = "";
      state.lastBubble = this.bubble;
    }
    this.text += chunk;
    this.bubble.innerHTML = markdown(this.text);
    scrollDown();
  }

  cut() {
    this.bubble?.classList.remove("streaming");
    this.bubble = null;
  }
}

const toolCards = new Map();

function toolStart({ id, name, args }) {
  const card = el("details", "tool running");
  const summary = el("summary");
  summary.append(el("span", "dot"), el("span", "tool-name", name), el("span", "tool-state", "running"));
  card.append(summary, el("pre", "", JSON.stringify(args ?? {}, null, 2)));
  toolCards.set(id, card);
  add(card);
}

function toolEnd({ id, ok, preview }) {
  const card = toolCards.get(id);
  if (!card) return;
  const declined = !ok && /^The user declined/.test(preview || "");
  const outcome = ok ? "ok" : declined ? "declined" : "failed";
  card.className = `tool ${outcome}`;
  card.querySelector(".tool-state").textContent = ok ? "done" : declined ? "declined" : "failed";
  if (preview) card.append(el("pre", "", preview));
  if (outcome === "failed") card.open = true; // an error is worth seeing; a decline was the user's own choice
}

function showApproval(payload) {
  state.approval = payload;
  for (const call of payload.calls) {
    const tool = toolCards.get(call.id);
    if (!tool) continue;
    tool.className = "tool waiting";
    tool.querySelector(".tool-state").textContent = "waiting for you";
  }
  const card = el("section", "approval");
  card.setAttribute("aria-label", "Needs your approval");
  card.append(el("h3", "", "Needs your OK"));
  for (const call of payload.calls) {
    card.append(el("p", "summary", call.summary || `Run ${call.name}.`));
    const details = el("details");
    details.append(el("summary", "", "Details"), el("pre", "", `${call.name}\n${JSON.stringify(call.args ?? {}, null, 2)}`));
    card.append(details);
  }
  const actions = el("div", "actions");
  const yes = el("button", "btn primary", "Approve");
  const no = el("button", "btn", "Decline");
  yes.type = no.type = "button";
  yes.onclick = () => decide(true, { voice: !!state.voice?.active });
  no.onclick = () => decide(false, { voice: !!state.voice?.active });
  actions.append(yes, no);
  card.append(actions);
  state.approvalCard = add(card);
  input.placeholder = "Or say what to change…";
  updateControls();
}

function settleApproval(word) {
  const card = state.approvalCard;
  if (card) {
    card.classList.add("done");
    card.querySelector("h3").textContent = word;
    card.querySelector(".actions")?.remove();
  }
  state.approval = null;
  state.approvalCard = null;
  input.placeholder = "Message…";
}

/** Under the message just spoken: how long after the speaker stopped the agent's first words played. */
function showLatency(ms) {
  const said = [...log.querySelectorAll(".msg.user")].pop();
  if (!said || said.nextElementSibling?.classList.contains("latency")) return;
  said.after(el("div", "latency heard-latency", `Answered aloud ${(ms / 1000).toFixed(2)} s after you stopped speaking`));
}

function hello() {
  const box = el("div", "hello");
  box.append(el("h2", "", state.config.name), el("p", "", state.config.description || "Ask me anything."));
  if (state.config.starters?.length) {
    const row = el("div", "starters");
    for (const text of state.config.starters) {
      const button = el("button", "starter", text);
      button.type = "button";
      button.onclick = () => sendMessage(text);
      row.append(button);
    }
    box.append(row);
  }
  log.append(box);
}

function setCaption(mode, detail) {
  caption.replaceChildren();
  if (mode === "listening") caption.append(detail ? el("span", "heard", detail) : el("span", "", "Listening…"));
  else if (detail) caption.append(el("span", "", detail));
}

function updateControls() {
  sendButton.disabled = state.busy || !input.value.trim();
  document.body.dataset.busy = state.busy ? "true" : "false";
}

function autosize() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
  updateControls();
}

// ------------------------------------------------------------------ running the agent

async function run(path, body) {
  state.busy = true;
  updateControls();
  const reply = new Reply();
  const voice = state.voice;
  showThinking();
  const onEvent = (event) => {
    if (event.type !== "tool_end") hideThinking();
    switch (event.type) {
      case "token":
        reply.append(event.text);
        voice?.feed(event.text);
        break;
      case "tool_start":
        reply.cut();
        toolStart(event);
        voice?.toolStarted(event);
        break;
      case "tool_end":
        toolEnd(event);
        if (!log.querySelector(".tool.running")) showThinking(); // the model reads the results next
        break;
      case "approval":
        reply.cut();
        showApproval(event);
        voice?.approval(event);
        break;
      case "error":
        reply.cut();
        notice(event.message);
        voice?.failed();
        break;
    }
  };
  try {
    await stream(path, body, onEvent);
  } catch (error) {
    reply.cut();
    notice(error.message || String(error));
    voice?.failed();
  } finally {
    hideThinking();
    reply.cut();
    state.busy = false;
    updateControls();
  }
  voice?.endReply();
}

async function sendMessage(text, { voice = false } = {}) {
  text = text.trim();
  if (!text || state.busy) return false;
  if (state.approval) settleApproval("Changed"); // words instead of an answer: the server declines and passes them on
  addUser(text);
  input.value = "";
  autosize();
  store.set("agent.thread", state.thread);
  await run("/chat", { thread_id: state.thread, message: text, voice });
  return true;
}

async function decide(approve, { voice = false } = {}) {
  if (!state.approval || state.busy) return;
  settleApproval(approve ? "Approved" : "Declined");
  await run("/resume", { thread_id: state.thread, approve, voice });
}

async function restore() {
  let data;
  try {
    data = await getJSON(`/threads/${state.thread}`);
  } catch (error) {
    notice(`Could not load the conversation: ${error.message}`);
    return;
  }
  for (const m of data.messages) {
    if (m.role === "user") addUser(m.text);
    else if (m.role === "assistant") addAgent(m.text);
    else {
      toolStart(m);
      if ("ok" in m) toolEnd(m);
    }
  }
  if (data.approval) showApproval(data.approval);
  if (!data.messages.length) hello();
}

function newChat() {
  state.voice?.stop();
  state.thread = newThreadId();
  store.set("agent.thread", "");
  state.approval = state.approvalCard = state.lastBubble = null;
  toolCards.clear();
  log.replaceChildren();
  input.placeholder = "Message…";
  hello();
  input.focus();
}

// ------------------------------------------------------------------ start

const VOICE_HINT = "Enter sends, Shift+Enter adds a line. Press the microphone to talk instead.";

async function setUpVoice(config) {
  const { createVoice } = await import("./voice.js");
  const voice = createVoice(config, {
    api,
    accessCode: () => state.code,
    send: (text) => sendMessage(text, { voice: true }),
    decide: (approve) => decide(approve, { voice: true }),
    pendingApproval: () => state.approval,
    busy: () => state.busy,
    mode(mode, detail) {
      dock.dataset.voice = mode || "";
      talk.setAttribute("aria-pressed", mode ? "true" : "false");
      talk.setAttribute("aria-label", mode === "speaking" ? "Interrupt and talk" : mode ? "End the voice conversation" : "Start a voice conversation");
      setCaption(mode, detail);
    },
    level: (value) => dock.style.setProperty("--level", value.toFixed(2)),
    latency: showLatency,
    error: notice,
    progress: (text) => {
      fineprint.textContent = text || VOICE_HINT; // model downloads on first use, then the usual hint
    },
  });
  if (!voice.supported) {
    fineprint.textContent = voice.reason;
    return null;
  }
  talk.hidden = false;
  talk.onclick = () => voice.toggle();
  fineprint.textContent = VOICE_HINT;
  return voice;
}

async function main() {
  const config = await (await fetch("/api/config")).json();
  state.config = config;
  document.title = config.name;
  document.querySelector(".name").textContent = config.name;
  document.querySelector(".about").textContent = config.description || "";
  document.querySelector(".mark").dataset.letter = (config.name || "A").trim().charAt(0).toUpperCase();
  if (config.locked && !state.code) await askForCode(false);
  if (config.voice) state.voice = await setUpVoice(config.voice);
  await restore();
}

input.addEventListener("input", autosize);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    sendMessage(input.value);
  }
});
$("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(input.value);
});
$("new-chat").addEventListener("click", newChat);

main().catch((error) => notice(`The agent did not load: ${error.message || error}`));
