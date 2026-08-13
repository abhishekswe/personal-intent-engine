<script>
  import { invoke } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import { onMount } from "svelte";

  import Icon from "./lib/Icon.svelte";
  import RecordingView from "./lib/RecordingView.svelte";
  import ModelManager from "./lib/ModelManager.svelte";
  import TranscriptionSettings from "./lib/TranscriptionSettings.svelte";
  import LLMSettings from "./lib/LLMSettings.svelte";
  import OutputSettings from "./lib/OutputSettings.svelte";
  import HotkeyRecorder from "./lib/HotkeyRecorder.svelte";
  import HistorySettings from "./lib/HistorySettings.svelte";
  import VocabularySettings from "./lib/VocabularySettings.svelte";
  import VocabularySync from "./lib/VocabularySync.svelte";
  import HistoryView from "./lib/HistoryView.svelte";

  let view = $state("record");

  let recState = $state("idle");
  let error = $state("");
  let outcome = $state(null);
  let llmResponse = $state("");
  let llmBusy = $state(false);
  let level = $state(0);
  let hasLevel = $state(false);

  let settings = $state({
    stt_model: "",
    silero_model: "",
    language: "auto",
    mode: "auto",
    provider: "echo",
    llm_api_url: "",
    llm_api_key: "",
    llm_model: "",
    hotkey_raw: "CmdOrCtrl+Shift+V",
    hotkey_optimized: "CmdOrCtrl+Shift+Space",
    paste_output: "transcript",
    history_limit: 10,
    deep_correct_ai: false,
    code_mode: false,
  });
  let saved = $state(false);
  let savedTimer;

  let models = $state([]);
  let downloads = $state({});

  // The window chrome sits on the left on macOS and the right on Windows.
  // Reserving space per platform instead of globally means neither one
  // carries the other's dead space.
  const ua = typeof navigator === "undefined" ? "" : navigator.userAgent;
  const isMac = ua.includes("Macintosh") || ua.includes("Mac OS");
  const isWin = ua.includes("Windows");

  async function loadModels() {
    try { models = await invoke("list_models"); }
    catch (e) { error = String(e); }
  }

  async function downloadModel(id) {
    error = "";
    downloads = { ...downloads, [id]: { received: 0, total: 0 } };
    try { await invoke("download_model", { id }); }
    catch (e) { error = String(e); }
  }

  async function selectModel(id) {
    try {
      await invoke("select_model", { id });
      settings = await invoke("get_settings");
      await loadModels();
    } catch (e) { error = String(e); }
  }

  async function deleteModel(id) {
    error = "";
    try {
      await invoke("delete_model", { id });
      await loadModels();
    } catch (e) { error = String(e); }
  }

  async function save() {
    try {
      await invoke("update_settings", { settings });
      saved = true;
      clearTimeout(savedTimer);
      savedTimer = setTimeout(() => { saved = false; }, 1500);
    } catch (e) { error = String(e); }
  }

  async function toggleRecording() {
    error = "";
    outcome = null;
    llmResponse = "";
    try {
      if (recState === "idle") {
        await invoke("start_recording");
      } else if (recState === "recording") {
        const result = await invoke("stop_recording");
        outcome = result;
      }
    } catch (e) { error = String(e); }
  }

  async function cancelRecording() {
    try { await invoke("cancel_recording"); }
    catch (e) { error = String(e); }
  }

  async function sendToLlm() {
    if (!outcome) return;
    llmBusy = true;
    llmResponse = "";
    try {
      llmResponse = await invoke("send_to_llm", {
        prompt: outcome.optimized_prompt,
      });
    } catch (e) { error = String(e); }
    finally { llmBusy = false; }
  }

  async function copyPrompt() {
    if (!outcome) return;
    try { await invoke("copy_to_clipboard", { text: outcome.optimized_prompt }); }
    catch (e) { error = String(e); }
  }

  async function onRecorrect() {
    if (!outcome) return;
    llmBusy = true;
    try {
      outcome = await invoke("recorrect_with_ai", { transcript: outcome.transcript });
    } catch (e) {
      error = String(e);
    } finally {
      llmBusy = false;
    }
  }

  const stateLabel = $derived(
    { idle: "Ready", recording: "Recording", decoding: "Transcribing…" }[recState] ?? recState
  );

  // The lexicon is a third of what PIE is, so it is a section of the book
  // rather than the sixth pane of a settings scroll.
  const TABS = [
    { id: "record",  label: "Record" },
    { id: "history", label: "History" },
    { id: "lexicon", label: "Lexicon" },
    { id: "models",  label: "Models" },
    { id: "setup",   label: "Setup" },
  ];

  onMount(() => {
    let unlisteners = [];
    let disposed = false;

    (async () => {
      try { settings = await invoke("get_settings"); }
      catch (e) { error = String(e); }
      await loadModels();

      const subs = await Promise.all([
        listen("pie://download", (event) => {
          const p = event.payload;
          if (p.done) {
            const next = { ...downloads };
            delete next[p.id];
            downloads = next;
            if (p.error) error = p.error;
            loadModels();
          } else {
            downloads = { ...downloads, [p.id]: { received: p.received, total: p.total } };
          }
        }),
        listen("pie://models-changed", () => loadModels()),
        listen("pie://state", (event) => {
          recState = event.payload;
          if (recState !== "recording") level = 0;
        }),
        listen("pie://level", (event) => {
          hasLevel = true;
          level = Math.max(0, Math.min(1, Number(event.payload) || 0));
        }),
        listen("pie://outcome", (event) => { outcome = event.payload; }),
        listen("pie://error", (event) => { error = event.payload; }),
      ]);

      if (disposed) { subs.forEach((u) => u()); return; }
      unlisteners = subs;
    })();

    return () => {
      disposed = true;
      unlisteners.forEach((u) => u());
      clearTimeout(savedTimer);
    };
  });
</script>

<svelte:window
  onkeydown={(e) => { if (e.key === "Escape" && recState === "recording") cancelRecording(); }}
/>

<!-- Thumb index. It already names the active section, so there is no running
     head repeating it. -->
<header class="head" class:is-mac={isMac} class:is-win={isWin}>
  {#if saved}
    <span class="saved-tag">Saved</span>
  {/if}
  <nav class="thumbs" aria-label="Sections">
    {#each TABS as tab}
      <button
        class="thumb"
        class:active={view === tab.id}
        aria-current={view === tab.id ? "page" : undefined}
        onclick={() => { view = tab.id; error = ""; }}
      >{tab.label}</button>
    {/each}
  </nav>
</header>

<main class="page">
  {#if error}
    <div class="error" role="alert">
      <span class="error-label">Error</span>
      <span class="error-text">{error}</span>
      <button class="error-x" onclick={() => { error = ""; }} aria-label="Dismiss error">
        <Icon name="close" size={13} />
      </button>
    </div>
  {/if}

  {#if view === "record"}
    <RecordingView
      {recState}
      {outcome}
      {llmResponse}
      {llmBusy}
      {level}
      {hasLevel}
      hotkey={settings.hotkey_optimized}
      {stateLabel}
      onToggle={toggleRecording}
      onCancel={cancelRecording}
      onSend={sendToLlm}
      onCopy={copyPrompt}
      {onRecorrect}
      onError={(e) => { error = e; }}
    />
  {:else if view === "history"}
    <HistoryView />
  {:else if view === "lexicon"}
    <VocabularySettings {settings} onSave={save} onError={(e) => { error = e; }} />
    <VocabularySync {settings} onSave={save} onError={(e) => { error = e; }} />
  {:else if view === "models"}
    <ModelManager
      {models}
      {downloads}
      {settings}
      onDownload={downloadModel}
      onSelect={selectModel}
      onDelete={deleteModel}
      onSave={save}
      onReloadModels={loadModels}
    />
  {:else if view === "setup"}
    <!-- The two global hotkeys are the product, so they open the section
         rather than sitting fifth in a flat list. -->
    <HotkeyRecorder {settings} onSave={save} onError={(e) => { error = e; }}
      field="hotkey_optimized" label="Optimized paste hotkey" defaultValue="CmdOrCtrl+Shift+Space" />
    <HotkeyRecorder {settings} onSave={save} onError={(e) => { error = e; }}
      field="hotkey_raw" label="Raw paste hotkey" defaultValue="CmdOrCtrl+Shift+V"
      showNote={false} />
    <TranscriptionSettings {settings} onSave={save} />
    <LLMSettings {settings} onSave={save} />
    <OutputSettings {settings} onSave={save} />
    <HistorySettings {settings} onSave={save} />
  {/if}
</main>
