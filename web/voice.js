// A voice conversation: listen, send what was said, speak the reply while it is still being written, listen again.
//
// Each direction has a provider (STT_PROVIDER / TTS_PROVIDER in .env):
//   device   on this device, the same as KIRA: Whisper hears (whisper.worker.js) and Supertonic speaks
//            (tts.worker.js). No audio or text leaves the browser and there is no per-use cost. The models download
//            once (about 77 MB and 262 MB) and are cached; the first reply waits while they start.
//   browser  the browser's own speech (Web Speech API): nothing to download, fastest, Chrome and Edge for listening.
//   openai / deepgram / elevenlabs   on the server (/api/stt, /api/tts), so their keys stay there.
//
// Whatever the provider, the reply is spoken a sentence at a time as tokens arrive, so the first words play while
// the model is still writing the rest; a short "One moment." covers a tool call; and each spoken reply shows the
// real delay from the end of the user's speech to its first word.

const YES = /\b(yes|yeah|yep|yup|sure|ok|okay|go ahead|do it|book it|confirm|confirmed|please do|sounds good|correct|haan|han|ha|ji)\b/i;
const NO = /\b(no|nope|nah|don'?t|do not|cancel|stop|wait|hold on|not now|nahi|nahin|mat)\b/i;
const CHANGE = /\b(but|instead|actually|change|make it|move|different|another|other|earlier|later)\b/i;
const DEVANAGARI = /[ऀ-ॿ]/; // Supertonic speaks English only; Hindi script goes to the device's voices
const FILLER = "One moment.";

const PAUSE_MS = 1000; // silence after speech that ends a turn (KIRA waits 1200 ms; a little less answers sooner)
const WAIT_MS = 8000; // how long to wait for speech to start
const LIMIT_MS = 30000; // longest single turn

/** What a spoken answer to an approval means: "yes", "no", or "say" (pass the words on, e.g. "yes but at 3"). */
export function answerOf(text) {
  const yes = YES.test(text);
  const no = NO.test(text);
  if (CHANGE.test(text)) return "say";
  if (yes && !no) return "yes";
  if (no && !yes && text.trim().split(/\s+/).length <= 4) return "no";
  return "say";
}

/** Text as it should sound: no Markdown, no links, and ranges read as "to". */
export function speakable(text) {
  return text
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/`([^`]*)`/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/https?:\/\/\S+/g, "")
    .replace(/\b(AM|PM|am|pm)\s*[–—-]\s*(?=\d)/g, "$1 to ")
    .replace(/(\d)\s*[–—]\s*(?=\d)/g, "$1 to ")
    .replace(/\s[–—]\s/g, ", ")
    .replace(/^\s*(?:[-*•]|\d+[.)])\s+/gm, "")
    .replace(/[*_#>|]+/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

/** Splits streamed text into sentences as soon as each one is complete. The first piece of a reply may be just its
 * first clause: a short piece is synthesised sooner, and the first words are what the listener waits for. */
export class Sentences {
  buffer = "";
  first = true;

  push(text) {
    this.buffer += text;
    const out = [];
    for (;;) {
      const match = this.buffer.match(/^[\s\S]*?(?:[.!?…](?=\s)|\n)/);
      if (match) {
        out.push(match[0]);
        this.buffer = this.buffer.slice(match[0].length);
        this.first = false;
        continue;
      }
      if (this.first) {
        const clause = this.buffer.slice(24).search(/[,;:](?=\s)/);
        if (clause >= 0) {
          out.push(this.buffer.slice(0, 24 + clause + 1));
          this.buffer = this.buffer.slice(24 + clause + 1);
          this.first = false;
          continue;
        }
      }
      if (this.buffer.length > 160) {
        // a long sentence: speak up to its last comma (or space) rather than wait for the end
        const cut = Math.max(this.buffer.lastIndexOf(", ", 150), this.buffer.lastIndexOf(" ", 150));
        if (cut > 40) {
          out.push(this.buffer.slice(0, cut + 1));
          this.buffer = this.buffer.slice(cut + 1);
          continue;
        }
      }
      return out;
    }
  }

  flush() {
    const rest = this.buffer;
    this.buffer = "";
    this.first = true; // what comes next starts a new stretch of speech
    return rest;
  }
}

// ------------------------------------------------------------------ the on-device models

/** Talks to a model worker: started on first use, replies matched to requests by id. */
function modelWorker(file, onProgress) {
  let worker = null;
  let next = 0;
  const jobs = new Map();
  const get = () => {
    if (worker) return worker;
    const w = new Worker(new URL(file, import.meta.url), { type: "module" });
    w.onmessage = ({ data }) => {
      if (data.type === "progress") onProgress(data.percent);
      else if (data.type === "ready") onProgress(null);
      else {
        jobs.get(data.id)?.(data.cancelled ? null : data);
        jobs.delete(data.id);
      }
    };
    w.onerror = (event) => {
      event.preventDefault(); // e.g. the model library could not be downloaded
      for (const done of jobs.values()) done({ error: "the model could not start" });
      jobs.clear();
      onProgress(null);
      worker = null;
      w.terminate();
    };
    return (worker = w);
  };
  return {
    load: () => get().postMessage({ type: "load" }),
    ask(message) {
      const id = next++;
      return new Promise((resolve) => {
        jobs.set(id, resolve);
        get().postMessage({ ...message, id });
      });
    },
    /** Drops every unanswered request: each resolves to null. */
    drop() {
      for (const done of jobs.values()) done(null);
      jobs.clear();
      worker?.postMessage({ type: "cancel" });
    },
  };
}

/** A recording as Whisper wants it: mono, 16 kHz. The browser resamples while decoding. */
async function decode16k(blob) {
  const context = new AudioContext({ sampleRate: 16000 });
  try {
    const audio = await context.decodeAudioData(await blob.arrayBuffer());
    if (audio.numberOfChannels === 1) return audio.getChannelData(0);
    const mono = new Float32Array(audio.length);
    for (let c = 0; c < audio.numberOfChannels; c++) {
      const channel = audio.getChannelData(c);
      for (let i = 0; i < channel.length; i++) mono[i] += channel[i] / audio.numberOfChannels;
    }
    return mono;
  } finally {
    context.close();
  }
}

// ------------------------------------------------------------------ speaking

function pickVoice(language, wanted) {
  const voices = window.speechSynthesis.getVoices();
  const lang = language.toLowerCase();
  const base = lang.split("-")[0];
  const pool = voices.filter((v) => v.lang.toLowerCase().replace("_", "-").startsWith(base));
  const same = (v) => v.lang.toLowerCase().replace("_", "-") === lang;
  const natural = (v) => /natural|neural|premium|enhanced/i.test(v.name);
  return (
    (wanted && voices.find((v) => v.name.toLowerCase().includes(wanted.toLowerCase()))) ||
    pool.find((v) => natural(v) && same(v)) ||
    pool.find(natural) ||
    pool.find((v) => /google/i.test(v.name) && same(v)) ||
    pool.find(same) ||
    pool[0] ||
    null
  );
}

/** Plays sentences one after another (Supertonic on this device, the browser's voices, or /api/tts) and says when
 * each starts. Supertonic is asked for every sentence as soon as it is queued, so the next clip is being made while
 * the current one plays. */
class Speaker {
  constructor(config, app, onStart, supertonic) {
    this.config = config;
    this.app = app;
    this.onStart = onStart;
    this.supertonic = supertonic;
    this.queue = [];
    this.playing = false;
    this.current = null;
    this.idle = [];
    this.player = null;
    this.deviceFailed = false; // after the model fails once (offline, say), the browser's voices take over
    if ("speechSynthesis" in window) {
      this.voice = pickVoice(config.language, config.browser_voice);
      window.speechSynthesis.onvoiceschanged = () => {
        this.voice = pickVoice(config.language, config.browser_voice);
      };
    }
  }

  /** Call inside a click: browsers start audio only after the user has interacted, and replies are spoken later. */
  unlock() {
    if (this.supertonic) {
      this.player ??= new AudioContext();
      this.player.resume();
      if (!this.deviceFailed) this.supertonic.load(); // downloads while the user is still talking
    }
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(new SpeechSynthesisUtterance("")); // unlocks speech inside this click (Safari)
    }
  }

  say(text) {
    const clean = speakable(text);
    if (!clean) return;
    const item = { text: clean };
    if (this.supertonic && !this.deviceFailed && !DEVANAGARI.test(clean)) item.clip = this.supertonic.ask({ type: "speak", text: clean });
    else if (!["browser", "device"].includes(this.config.tts)) item.audio = this.prefetch(clean);
    this.queue.push(item);
    if (!this.playing) this.drain();
  }

  prefetch(text) {
    const params = new URLSearchParams({ text });
    const code = this.app.accessCode();
    if (code) params.set("code", code);
    const audio = new Audio(`/api/tts?${params}`); // starts downloading now, ready when the sentence before ends
    audio.preload = "auto";
    return audio;
  }

  async drain() {
    this.playing = true;
    while (this.queue.length) {
      const item = this.queue.shift();
      try {
        if (item.clip) {
          const reply = await item.clip;
          if (!reply) continue; // cancelled
          if (reply.error) {
            this.deviceFailed = true;
            console.warn("Supertonic couldn't speak; using the browser's voices instead.", reply.error);
            await this.playBrowser(item.text);
          } else await this.playClip(reply.audio, reply.rate);
        } else if (item.audio) await this.playAudio(item.audio);
        else await this.playBrowser(item.text);
      } catch {
        if ((item.audio || item.clip) && "speechSynthesis" in window) await this.playBrowser(item.text).catch(() => {});
      }
    }
    this.playing = false;
    this.current = null;
    const waiting = this.idle;
    this.idle = [];
    waiting.forEach((resolve) => resolve());
  }

  playClip(samples, rate) {
    return new Promise((resolve) => {
      const out = this.player;
      const buffer = out.createBuffer(1, samples.length, rate);
      buffer.copyToChannel(samples, 0);
      const source = out.createBufferSource();
      source.buffer = buffer;
      source.connect(out.destination);
      source.onended = () => resolve();
      this.current = {
        stop: () => {
          try {
            source.stop();
          } catch {
            /* already ended */
          }
          resolve();
        },
      };
      source.start();
      this.onStart();
    });
  }

  playAudio(audio) {
    return new Promise((resolve, reject) => {
      this.current = { stop: () => { audio.pause(); resolve(); } };
      audio.onplaying = () => this.onStart();
      audio.onended = () => resolve();
      audio.onerror = () => reject(new Error("speech failed"));
      audio.play().catch(reject);
    });
  }

  playBrowser(text) {
    if (!("speechSynthesis" in window)) return Promise.resolve();
    return new Promise((resolve) => {
      const utterance = new SpeechSynthesisUtterance(text);
      const voice = DEVANAGARI.test(text) ? pickVoice("hi-IN") : this.voice;
      if (voice) utterance.voice = voice;
      utterance.lang = voice?.lang || (DEVANAGARI.test(text) ? "hi-IN" : this.config.language);
      utterance.rate = 1.05;
      utterance.onstart = () => this.onStart();
      utterance.onend = utterance.onerror = () => resolve();
      this.current = { stop: () => { window.speechSynthesis.cancel(); resolve(); } };
      window.speechSynthesis.speak(utterance);
    });
  }

  /** Resolves once everything queued has been spoken (at once if nothing is). */
  whenIdle() {
    return this.playing ? new Promise((resolve) => this.idle.push(resolve)) : Promise.resolve();
  }

  cancel() {
    this.queue = [];
    this.supertonic?.drop();
    this.current?.stop();
    if ("speechSynthesis" in window) window.speechSynthesis.cancel();
  }
}

// ------------------------------------------------------------------ listening

/** One turn in the browser (Web Speech API). Calls done(text, speechEndedAt) once; "" when nothing was heard. */
function listenInBrowser(language, { interim, done, failed }) {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const recognition = new Recognition();
  recognition.lang = language;
  recognition.interimResults = true;
  recognition.continuous = false;
  let finished = false;
  let heard = "";
  let speechEnd = 0;
  const finish = (text) => {
    if (finished) return;
    finished = true;
    done(text.trim(), speechEnd || performance.now());
  };
  recognition.onspeechend = () => {
    speechEnd = performance.now();
  };
  recognition.onresult = (event) => {
    let partial = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const result = event.results[i];
      if (result.isFinal) {
        heard += result[0].transcript;
        finish(heard); // hand it over now rather than waiting for the recognizer to wind down
        recognition.stop();
        return;
      }
      partial += result[0].transcript;
    }
    interim(`${heard} ${partial}`.trim());
  };
  recognition.onerror = (event) => {
    if (event.error === "no-speech" || event.error === "aborted") return;
    finished = true;
    failed(event.error === "not-allowed" ? "The microphone is blocked. Allow it in the address bar, then try again." : `Speech recognition failed (${event.error}).`);
  };
  recognition.onend = () => finish(heard);
  recognition.start();
  return { cancel: () => { finished = true; recognition.abort(); } };
}

/** Records one turn: it ends after a pause once the speaker has talked, or after WAIT_MS of silence. The speech
 * threshold follows the room's noise floor (as in KIRA), so a fan doesn't count as speech and a quiet microphone
 * still does. Calls done(blob or null, speechEndedAt). */
function recordTurn(stream, rms, { done }) {
  const types = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"];
  const mimeType = types.find((t) => window.MediaRecorder.isTypeSupported?.(t)) || "";
  const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  const chunks = [];
  const started = performance.now();
  let spoke = false;
  let lastSound = started;
  let floor = 0.01;
  let cancelled = false;
  const finish = () => {
    clearInterval(timer);
    if (recorder.state !== "inactive") recorder.stop();
  };
  const timer = setInterval(() => {
    const level = rms();
    const now = performance.now();
    floor = Math.min(floor * 1.005, Math.max(level, 0.002));
    if (level > Math.max(0.012, floor * 3)) {
      spoke = true;
      lastSound = now;
    }
    if (now - started > LIMIT_MS || (spoke ? now - lastSound > PAUSE_MS : now - started > WAIT_MS)) finish();
  }, 50);
  recorder.ondataavailable = (event) => event.data.size && chunks.push(event.data);
  recorder.onstop = () => {
    if (cancelled) return;
    const heard = spoke && chunks.length ? new Blob(chunks, { type: recorder.mimeType || "audio/webm" }) : null;
    done(heard, spoke ? lastSound : performance.now());
  };
  recorder.start(250);
  return {
    cancel() {
      cancelled = true;
      finish();
    },
  };
}

/** One turn heard on this device: record, then Whisper turns it into text. */
function listenOnDevice(stream, rms, whisper, language, { done, failed, transcribing }) {
  let cancelled = false;
  const turn = recordTurn(stream, rms, {
    async done(blob, endedAt) {
      if (!blob) return done("", endedAt);
      transcribing();
      try {
        const reply = await whisper.ask({ type: "transcribe", audio: await decode16k(blob), language });
        if (cancelled || !reply) return;
        if (reply.error) throw new Error(reply.error);
        done(String(reply.text || "").trim(), endedAt);
      } catch (error) {
        if (!cancelled) failed(`Speech recognition failed: ${error.message}. Try again.`);
      }
    },
  });
  return {
    cancel() {
      cancelled = true;
      turn.cancel();
    },
  };
}

/** One turn heard on the server: record, then /api/stt turns it into text. */
function listenWithServer(stream, rms, app, { done, failed, transcribing }) {
  let cancelled = false;
  const turn = recordTurn(stream, rms, {
    async done(blob, endedAt) {
      if (!blob) return done("", endedAt);
      transcribing();
      try {
        const res = await app.api("/stt", { method: "POST", headers: { "Content-Type": blob.type }, body: blob });
        const body = await res.json();
        if (!res.ok) throw new Error(body.detail || `${res.status}`);
        if (!cancelled) done(String(body.text || "").trim(), endedAt);
      } catch (error) {
        if (!cancelled) failed(`Speech to text failed: ${error.message}`);
      }
    },
  });
  return {
    cancel() {
      cancelled = true;
      turn.cancel();
    },
  };
}

// ------------------------------------------------------------------ the conversation

export function createVoice(config, app) {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const hearsOnDevice = config.stt === "device";
  const speaksOnDevice = config.tts === "device";
  const recording = !!(navigator.mediaDevices?.getUserMedia && window.MediaRecorder);
  const canListen = config.stt === "browser" ? !!Recognition : recording && (!hearsOnDevice || typeof Worker !== "undefined");
  const canSpeak = config.tts === "browser" ? "speechSynthesis" in window : true;

  // Model downloads (first use only), shown under the composer
  const downloads = { speech: null, voice: null };
  const showDownloads = () => {
    const [kind, percent] = Object.entries(downloads).find(([, p]) => p !== null) ?? [];
    if (kind === undefined) return app.progress("");
    app.progress(percent < 100 ? `Downloading the ${kind} model, first time only: ${Math.round(percent)}%` : `Starting the ${kind} model…`);
  };
  const track = (kind) => (percent) => {
    downloads[kind] = percent ?? null;
    showDownloads();
  };
  const whisper = hearsOnDevice ? modelWorker("./whisper.worker.js", track("speech")) : null;
  const supertonic = speaksOnDevice ? modelWorker("./tts.worker.js", track("voice")) : null;
  const whisperLanguage = (config.language || "en").split("-")[0].toLowerCase();

  // A returning visitor has the models cached: start them now, so the first spoken turn doesn't wait for them.
  // A first-time visitor downloads them only on pressing the microphone.
  if (whisper || supertonic) {
    window.caches
      ?.has("transformers-cache")
      .then((cached) => {
        if (!cached) return;
        whisper?.load();
        supertonic?.load();
      })
      .catch(() => {});
  }

  let active = false;
  let mode = null;
  let listening = null;
  let stream = null;
  let meter = null;
  let heardAt = 0; // when the user stopped speaking; cleared once the reply's first word plays
  let spokeThisReply = false;
  let muted = false; // interrupted: the rest of this reply is shown, not spoken
  const sentences = new Sentences();
  const speaker = new Speaker(
    config,
    app,
    () => {
      if (heardAt) {
        app.latency(performance.now() - heardAt);
        heardAt = 0;
      }
      setMode("speaking", "Speaking. Press the button to interrupt.");
    },
    supertonic,
  );

  function setMode(next, detail = "") {
    mode = next;
    app.mode(next, detail);
  }

  async function openMicrophone() {
    if (stream) return;
    stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
    const context = new AudioContext();
    const analyser = context.createAnalyser();
    analyser.fftSize = 1024;
    context.createMediaStreamSource(stream).connect(analyser);
    const samples = new Float32Array(analyser.fftSize);
    meter = { context, rms: 0, frame: 0 };
    const tick = () => {
      if (!meter) return;
      analyser.getFloatTimeDomainData(samples);
      let sum = 0;
      for (const s of samples) sum += s * s;
      meter.rms = Math.sqrt(sum / samples.length);
      app.level(mode === "listening" || mode === "waiting" ? Math.min(1, meter.rms * 10) : 0);
      meter.frame = requestAnimationFrame(tick);
    };
    tick();
  }

  function closeMicrophone() {
    if (meter) {
      cancelAnimationFrame(meter.frame);
      meter.context.close();
      meter = null;
    }
    stream?.getTracks().forEach((t) => t.stop());
    stream = null;
    app.level(0);
  }

  function listen() {
    if (!active) return;
    const approval = app.pendingApproval();
    setMode(approval ? "waiting" : "listening", approval ? "Say yes to go ahead, or no. You can also say what to change." : "");
    const handlers = {
      interim: (text) => setMode(approval ? "waiting" : "listening", text),
      transcribing: () => setMode("thinking", "Turning your speech into text…"),
      done: (text, endedAt) => {
        listening = null;
        if (!active) return;
        if (!text) {
          stop("Didn't catch anything. Press the microphone to talk again.");
          return;
        }
        heardAt = endedAt;
        route(text);
      },
      failed: (message) => {
        listening = null;
        stop();
        app.error(message);
      },
    };
    const rms = () => meter?.rms ?? 0;
    if (config.stt === "browser") listening = listenInBrowser(config.language, handlers);
    else if (hearsOnDevice) listening = listenOnDevice(stream, rms, whisper, whisperLanguage, handlers);
    else listening = listenWithServer(stream, rms, app, handlers);
  }

  function route(text) {
    spokeThisReply = false;
    muted = false;
    setMode("thinking", `“${text}”`);
    const answer = app.pendingApproval() ? answerOf(text) : "say";
    if (answer === "say") app.send(text); // while an approval waits, the server declines it and passes the words on
    else app.decide(answer === "yes");
  }

  async function start() {
    if (active) return;
    active = true;
    speaker.unlock(); // before any await: it has to run inside the click
    whisper?.load();
    try {
      await openMicrophone(); // also asks for permission up front, and drives the level ring
    } catch {
      active = false;
      app.error("The microphone is blocked. Allow it in the address bar, then try again.");
      return;
    }
    if (app.busy()) setMode("thinking", "Waiting for the reply…");
    else if (app.pendingApproval()) speakApproval(app.pendingApproval()).then(listen);
    else listen();
  }

  function stop(detail) {
    active = false;
    listening?.cancel();
    listening = null;
    speaker.cancel();
    sentences.flush();
    closeMicrophone();
    setMode(null);
    if (detail) app.mode(null, detail);
  }

  function speakApproval(payload) {
    const summary = payload.calls.map((c) => c.summary).join(" ");
    speaker.say(`${summary} Shall I go ahead?`);
    return speaker.whenIdle();
  }

  return {
    supported: canListen && canSpeak,
    reason: !canListen
      ? config.stt === "browser"
        ? "Voice needs Chrome or Edge here (or another STT_PROVIDER). Typing works everywhere."
        : "This browser can't record audio here; voice needs HTTPS or localhost. Typing works."
      : "This browser cannot speak replies aloud. Typing works.",
    get active() {
      return active;
    },
    toggle() {
      if (!active) return start();
      if (mode === "speaking") {
        speaker.cancel(); // interrupt: stop talking; the rest of the reply stays on screen
        sentences.flush();
        muted = true;
        if (app.busy()) setMode("thinking", "Finishing the reply quietly…");
        else listen();
        return;
      }
      stop();
    },
    stop,
    feed(text) {
      if (!active) return;
      for (const sentence of sentences.push(text)) {
        spokeThisReply = true;
        if (!muted) speaker.say(sentence);
      }
    },
    toolStarted() {
      if (!active || muted) return;
      const rest = sentences.flush(); // "Let me check your calendar" arrives without a full stop before the call
      if (rest.trim()) speaker.say(rest);
      else if (!spokeThisReply) speaker.say(FILLER); // covers the tool's wait; counts as the reply's first words
      spokeThisReply = true;
    },
    approval(payload) {
      if (!active) return;
      const rest = sentences.flush();
      if (rest.trim() && !muted) speaker.say(rest);
      muted = false; // the approval question is always spoken
      speakApproval(payload);
    },
    failed() {
      if (active) speaker.say("Something went wrong. The details are on screen.");
    },
    async endReply() {
      if (!active) return;
      const rest = sentences.flush();
      if (rest.trim() && !muted) speaker.say(rest);
      await speaker.whenIdle();
      if (active && !listening && !app.busy()) listen();
    },
  };
}
