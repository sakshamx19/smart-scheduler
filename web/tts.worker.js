// Text to speech on this device (TTS_PROVIDER=device): Supertone's Supertonic (open weights) through
// transformers.js, off the main thread. The same model, voice and settings KIRA uses. About 262 MB to download once,
// then the browser caches it. English only: voice.js reads Devanagari with the device's own voices.
//
// Speed depends on threads: the server sends the cross-origin isolation headers that let WebAssembly use several
// (about 1.3–1.8 s for a 4-second sentence on a 12-core laptop); without them (Safari) the GPU is used instead.
// Messages: voice.js.
import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0/dist/transformers.min.js";

const MODEL = "onnx-community/Supertonic-TTS-ONNX";
const VOICE = `https://huggingface.co/${MODEL}/resolve/main/voices/F1.bin`;
const STEPS = 5; // the authors' default; 2 is about a third faster and slightly rougher

let tts = null;
let queue = [];
let running = false;

async function load() {
  const progress_callback = (info) => {
    if (info.status === "progress_total") self.postMessage({ type: "progress", percent: info.progress });
  };
  // Threads make WebAssembly the fastest choice; without them the GPU wins.
  const useGpu = !self.crossOriginIsolated && !!(await navigator.gpu?.requestAdapter().catch(() => null));
  return pipeline("text-to-speech", MODEL, { device: useGpu ? "webgpu" : "wasm", dtype: "fp32", progress_callback });
}

function model() {
  if (!tts) {
    tts = load();
    tts.then(
      () => self.postMessage({ type: "ready" }),
      () => {
        tts = null; // let the next request try again, after a network error say
        self.postMessage({ type: "ready" });
      },
    );
  }
  return tts;
}

// One sentence at a time, in order: voice.js asks for each sentence as soon as the agent has written it, and plays
// each clip as it arrives.
async function drain() {
  if (running) return;
  running = true;
  while (queue.length) {
    const job = queue.shift();
    try {
      const speaker = await model();
      const clip = await speaker(job.text, { speaker_embeddings: VOICE, num_inference_steps: STEPS });
      self.postMessage({ id: job.id, audio: clip.audio, rate: clip.sampling_rate }, [clip.audio.buffer]);
    } catch (error) {
      self.postMessage({ id: job.id, error: error instanceof Error ? error.message : String(error) });
    }
  }
  running = false;
}

self.onmessage = ({ data }) => {
  if (data.type === "load") model().catch(() => {});
  else if (data.type === "cancel") {
    for (const job of queue) self.postMessage({ id: job.id, cancelled: true });
    queue = [];
  } else {
    queue.push({ id: data.id, text: data.text });
    drain();
  }
};
