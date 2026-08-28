<script>
  // The lexicon proper: the user's own heard→canonical corrections (the
  // "pronunciation.json" dictionary), the terms PIE learned on its own, and
  // the switches that govern how it learns.
  //
  // Corrections lead the section because they are the thing PIE keeps: an
  // entry taught once is matched instantly and offline forever after.
  import { invoke } from "@tauri-apps/api/core";
  import Icon from "./Icon.svelte";

  let { settings, onSave, onError } = $props();

  let corrections = $state([]);
  let heard = $state("");
  let canonical = $state("");
  let learnedCount = $state(0);

  async function refresh() {
    try {
      corrections = await invoke("list_corrections");
      learnedCount = await invoke("get_learned_vocab_count");
    }
    catch (e) { onError(String(e)); }
  }
  refresh();

  async function resetLearned() {
    try {
      await invoke("reset_learned_vocab");
      await refresh();
    } catch (e) { onError(String(e)); }
  }

  async function add() {
    if (!heard.trim() || !canonical.trim()) return;
    try {
      await invoke("add_correction", { heard: heard.trim(), canonical: canonical.trim() });
      heard = "";
      canonical = "";
      await refresh();
    } catch (e) { onError(String(e)); }
  }

  async function remove(h) {
    try {
      await invoke("delete_correction", { heard: h });
      await refresh();
    } catch (e) { onError(String(e)); }
  }
</script>

<section class="leaf">
  <div class="leaf-head">
    <span class="leaf-label">Your corrections</span>
    <span class="leaf-rule"></span>
    <span class="leaf-meta">{corrections.length}</span>
  </div>

  <div class="lex-add">
    <input class="text-input" placeholder="heard (e.g. next jazz)" bind:value={heard} />
    <span class="mark-arrow"><Icon name="arrow" size={13} /></span>
    <input class="text-input" placeholder="correct (e.g. Next.js)" bind:value={canonical} />
    <button class="btn sm" onclick={add} aria-label="Add correction">Add</button>
  </div>

  {#if corrections.length}
    <ul class="lex-list">
      {#each corrections as c (c.heard)}
        <li>
          <span class="lex-heard">{c.heard}</span>
          <span class="mark-arrow"><Icon name="arrow" size={12} /></span>
          <span class="lex-canon">{c.canonical}</span>
          <button
            class="text-btn"
            onclick={() => remove(c.heard)}
            aria-label={`Delete correction for ${c.heard}`}
          >Delete</button>
        </li>
      {/each}
    </ul>
  {:else}
    <p class="note">No custom corrections yet. Add one above, or save one from a result.</p>
  {/if}
</section>

<section class="leaf">
  <div class="leaf-head">
    <span class="leaf-label">Learned vocabulary</span>
    <span class="leaf-rule"></span>
    <span class="leaf-meta">{learnedCount}</span>
  </div>
  <p class="note">{learnedCount} terms learned automatically</p>
  <div class="actions">
    <button class="btn ghost sm" onclick={resetLearned} aria-label="Reset learned vocabulary">
      Reset learned
    </button>
  </div>
</section>

<section class="leaf">
  <div class="leaf-head">
    <span class="leaf-label">How PIE learns</span>
    <span class="leaf-rule"></span>
  </div>

  <div class="field">
    <label class="check-row">
      <input
        type="checkbox"
        bind:checked={settings.deep_correct_ai}
        onchange={onSave}
      />
      <span>
        <span class="field-label">Deep-correct with AI</span>
        <span class="note">
          Use the configured LLM to fix garbled terms the dictionary misses.
          Slower, and uses your provider.
        </span>
      </span>
    </label>
  </div>

  <div class="field">
    <label class="check-row">
      <input
        type="checkbox"
        bind:checked={settings.background_mining}
        onchange={onSave}
      />
      <span>
        <span class="field-label">Background learning</span>
        <span class="note">
          Mine new pronunciation corrections from your transcripts using the
          configured LLM, in the background. Takes effect on restart.
        </span>
      </span>
    </label>
  </div>

  <div class="field">
    <label class="check-row">
      <input
        type="checkbox"
        bind:checked={settings.code_mode}
        onchange={onSave}
      />
      <span>
        <span class="field-label">Code mode</span>
        <span class="note">
          Translate spoken code (“console dot log” → console.log() ). Only turn
          on while dictating code.
        </span>
      </span>
    </label>
  </div>

  <div class="field">
    <label class="check-row">
      <input
        type="checkbox"
        bind:checked={settings.enhance_with_ai}
        onchange={onSave}
      />
      <span>
        <span class="field-label">Enhance with AI</span>
        <span class="note">
          Off (default) = instant voice-to-text: transcribe + word correction,
          no model call. On = rewrite dictation into a structured prompt using
          your configured LLM. Slower — only turn on when you want the AI rewrite.
        </span>
      </span>
    </label>
  </div>
</section>
