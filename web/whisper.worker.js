// Speech to text on this device (STT_PROVIDER=device): OpenAI's Whisper (base, multilingual) through transformers.js,
// off the main thread. The same model and settings KIRA uses. With a GPU that supports fp16 it runs on WebGPU (about
// 146 MB to download); otherwise on WebAssembly, 8-bit (about 77 MB). The browser caches the files after the first
// download, and no audio leaves the device. Messages: voice.js.
import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0/dist/transformers.min.js";

const MODEL = "onnx-community/whisper-base";

let asr = null;

async function load() {
  const progress_callback = (info) => {
    if (info.status === "progress_total") self.postMessage({ type: "progress", percent: info.progress });
  };
  const adapter = await navigator.gpu?.requestAdapter().catch(() => null);
  if (adapter?.features.has("shader-f16")) {
    try {
      return await pipeline("automatic-speech-recognition", MODEL, { device: "webgpu", dtype: "fp16", progress_callback });
    } catch (error) {
      console.warn("Whisper couldn't start on WebGPU; using WebAssembly instead.", error);
    }
  }
  return pipeline("automatic-speech-recognition", MODEL, { device: "wasm", dtype: "q8", progress_callback });
}

function model() {
  if (!asr) {
    asr = load();
    asr.then(
      () => self.postMessage({ type: "ready" }),
      () => {
        asr = null; // let the next request try again, after a network error say
        self.postMessage({ type: "ready" });
      },
    );
  }
  return asr;
}

self.onmessage = async ({ data }) => {
  if (data.type === "load") {
    model().catch(() => {});
    return;
  }
  try {
    const transcriber = await model();
    const options = { task: "transcribe", chunk_length_s: 30, stride_length_s: 5 };
    let output;
    try {
      output = await transcriber(data.audio, { ...options, language: data.language });
    } catch (error) {
      // A language Whisper doesn't know: let it detect the language instead.
      if (!(error instanceof Error && error.message.startsWith("Language"))) throw error;
      output = await transcriber(data.audio, options);
    }
    const text = (Array.isArray(output) ? output[0] : output).text.trim();
    self.postMessage({ id: data.id, text });
  } catch (error) {
    self.postMessage({ id: data.id, error: error instanceof Error ? error.message : String(error) });
  }
};
