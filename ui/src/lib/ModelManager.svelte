<script>
  import Icon from "./Icon.svelte";

  let { models, downloads, settings, onDownload, onSelect, onDelete, onSave, onReloadModels } = $props();
  let showCustomPaths = $state(false);
</script>

{#snippet modelRow(m)}
  <li class="row-item" class:selected={m.selected}>
    <span class="row-main">
      <span class="row-name">
        {m.name}
        {#if m.selected}<span class="in-use">In use</span>{/if}
      </span>
      <span class="row-desc">{m.description} · {m.size_mb} MB</span>
    </span>
    <span class="row-actions">
      {#if downloads[m.id]}
        {@const d = downloads[m.id]}
        {@const p = d.total ? Math.round((d.received / d.total) * 100) : 0}
        <span class="progress" title="{p}%">
          <span class="progress-bar" style="--fill:{p / 100}"></span>
        </span>
        <span class="progress-pct">{p}%</span>
      {:else if !m.downloaded}
        <button class="btn sm" onclick={() => onDownload(m.id)} aria-label="Download {m.name}">Download</button>
      {:else if m.selected}
        <button class="btn ghost sm" disabled>Selected</button>
      {:else}
        <button class="btn sm" onclick={() => onSelect(m.id)} aria-label="Use {m.name}">Use</button>
        <button
          class="btn ghost sm icon"
          onclick={() => onDelete(m.id)}
          aria-label="Delete {m.name}"
          title="Delete model"
        ><Icon name="trash" size={13} /></button>
      {/if}
    </span>
  </li>
{/snippet}

<section class="leaf">
  <div class="leaf-head">
    <span class="leaf-label">Speech to text</span>
    <span class="leaf-rule"></span>
  </div>
  <ul class="list">
    {#each models.filter((m) => m.kind === "stt") as m}
      {@render modelRow(m)}
    {/each}
  </ul>
</section>

<section class="leaf">
  <div class="leaf-head">
    <span class="leaf-label">Voice detection</span>
    <span class="leaf-rule"></span>
  </div>
  <ul class="list">
    {#each models.filter((m) => m.kind === "vad") as m}
      {@render modelRow(m)}
    {/each}
  </ul>
  <p class="note">Optional. Trims silence so only speech is transcribed.</p>
</section>

<details class="disclosure" bind:open={showCustomPaths}>
  <summary>
    <Icon name="chevron-right" size={11} />
    Custom model paths
  </summary>
  <div class="disclosure-body">
    <div class="field">
      <label for="stt">Speech-to-text model id</label>
      <input
        id="stt"
        bind:value={settings.stt_model}
        onblur={() => { onSave(); onReloadModels(); }}
        placeholder="moonshine-base"
      />
    </div>
    <div class="field">
      <label for="silero">Voice detection model path</label>
      <input
        id="silero"
        bind:value={settings.silero_model}
        onblur={() => { onSave(); onReloadModels(); }}
        placeholder="~/.cache/pie/models/silero_vad_v4.onnx"
      />
    </div>
    <p class="note">Override the catalog with your own model files.</p>
  </div>
</details>
