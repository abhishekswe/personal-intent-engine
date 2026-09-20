<script>
  import { invoke } from "@tauri-apps/api/core";
  import { onDestroy } from "svelte";
  import { keycaps } from "./keycaps.js";

  let { settings, onSave, onError, defaultValue } = $props();
  let capturing = $state(false);

  const MODIFIERS = new Set([
    "MetaLeft", "MetaRight", "ControlLeft", "ControlRight",
    "AltLeft", "AltRight", "ShiftLeft", "ShiftRight",
  ]);

  async function beginCapture() {
    onError("");
    try {
      await invoke("set_hotkey_active", { active: false });
      capturing = true;
    } catch (e) {
      onError(String(e));
    }
  }

  async function finish(value) {
    capturing = false;
    if (value === null) {
      await invoke("set_hotkey_active", { active: true }).catch(() => {});
      return;
    }
    settings.hotkey = value;
    await onSave();
  }

  function modifierNames(e) {
    const names = [];
    if (e.metaKey) names.push("Command");
    if (e.ctrlKey) names.push("Control");
    if (e.altKey) names.push("Alt");
    if (e.shiftKey) names.push("Shift");
    return names;
  }

  function onKeydown(e) {
    if (!capturing) return;
    e.preventDefault();
    e.stopPropagation();
    if (e.code === "Escape") return finish(null);

    if (MODIFIERS.has(e.code)) return;
    if (!e.code || e.code === "Unidentified") return;

    const modifiers = modifierNames(e);
    const functionKey = /^F\d{1,2}$/.test(e.code);
    if (modifiers.length === 0 && !functionKey) return;
    finish([...modifiers, e.code].join("+"));
  }

  function reset() {
    settings.hotkey = defaultValue;
    onSave();
  }

  onDestroy(() => {
    if (capturing) invoke("set_hotkey_active", { active: true }).catch(() => {});
  });
</script>

<svelte:window onkeydown={onKeydown} />

<section class="leaf">
  <div class="leaf-head">
    <span class="leaf-label">Dictation shortcut</span>
    <span class="leaf-rule"></span>
  </div>

  <div class="field">
    <div class="hotkey-row">
      <div class="hotkey-display" class:capturing>
        {#if capturing}
          <span class="capture-hint">Press your shortcut…</span>
        {:else}
          <span class="keys">
            {#each keycaps(settings.hotkey) as cap}<kbd>{cap}</kbd>{/each}
          </span>
        {/if}
      </div>
      {#if capturing}
        <button class="btn ghost sm" onclick={() => finish(null)}>Cancel</button>
      {:else}
        <button class="btn sm" onclick={beginCapture}>Change</button>
      {/if}
    </div>

    {#if capturing}
      <p class="note">Press a modifier plus another key, such as Ctrl+Space.</p>
    {:else}
      <div class="hotkey-actions">
        <button class="text-btn" onclick={reset}>Reset to default</button>
      </div>
      <p class="note">
        Press once in any app to start recording. Press it again to finish and paste.
      </p>
    {/if}
  </div>
</section>
