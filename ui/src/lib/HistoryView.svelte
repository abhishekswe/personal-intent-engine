<script>
  // Past recordings, read as a concordance: what you said, when, and the
  // operations available on each entry.
  import { invoke } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import { onMount } from "svelte";

  let entries = $state([]);
  let query = $state("");
  let error = $state("");
  let searchTimer;

  async function refresh() {
    error = "";
    try {
      entries = await invoke("list_history", { query: query || null });
    } catch (e) { error = String(e); }
  }

  function onSearch() {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(refresh, 150);
  }

  async function copy(text) {
    error = "";
    try { await invoke("copy_to_clipboard", { text }); }
    catch (e) { error = String(e); }
  }

  async function paste(id) {
    error = "";
    try { await invoke("paste_history_entry", { id }); }
    catch (e) { error = String(e); }
  }

  async function remove(id) {
    error = "";
    try { await invoke("delete_history_entry", { id }); await refresh(); }
    catch (e) { error = String(e); }
  }

  async function clearAll() {
    if (!confirm("Delete all history?")) return;
    error = "";
    try { await invoke("clear_history"); await refresh(); }
    catch (e) { error = String(e); }
  }

  function relTime(unixSeconds) {
    const s = Math.max(0, Math.floor(Date.now() / 1000 - unixSeconds));
    if (s < 60) return "just now";
    if (s < 3600) return `${Math.floor(s / 60)}m ago`;
    if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
    return `${Math.floor(s / 86400)}d ago`;
  }

  onMount(() => {
    refresh();
    let unlisten;
    let disposed = false;
    listen("pie://history-changed", () => refresh()).then((u) => {
      if (disposed) u(); else unlisten = u;
    });
    return () => {
      disposed = true;
      if (unlisten) unlisten();
      clearTimeout(searchTimer);
    };
  });
</script>

<div class="field search">
  <label for="hist-search" class="field-label">Search</label>
  <input
    id="hist-search"
    class="text-input"
    placeholder="Search transcripts…"
    bind:value={query}
    oninput={onSearch}
  />
</div>

{#if error}
  <p class="note is-proof">{error}</p>
{/if}

{#if entries.length === 0}
  <div class="empty">
    <p class="empty-lead">No recordings yet.</p>
    <p class="note">Hold ⌥ or record to start.</p>
  </div>
{:else}
  <section class="leaf">
    <div class="leaf-head">
      <span class="leaf-label">Recordings</span>
      <span class="leaf-rule"></span>
      <span class="leaf-meta">{entries.length}</span>
    </div>

    {#each entries as e (e.id)}
      <article class="hist-item">
        <p class="hist-text">{e.transcript}</p>
        <div class="hist-meta">
          <span class="hist-time">{relTime(e.created_at)}</span>
          <div class="hist-actions">
            <button class="text-btn" onclick={() => copy(e.transcript)}>Copy</button>
            <button class="text-btn" onclick={() => paste(e.id)}>Paste</button>
            <button class="text-btn danger" onclick={() => remove(e.id)}>Delete</button>
          </div>
        </div>
      </article>
    {/each}

    <div class="actions">
      <button class="btn danger sm" onclick={clearAll}>Clear all</button>
    </div>
  </section>
{/if}
