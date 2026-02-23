// Archived duplicate — no-op stub. See archived_builder_duplicates/guided_builder_v2.broken.js
/* This file was a duplicated/broken variant of the guided builder.
   It has been replaced with a no-op to avoid duplicate runtime wiring.
*/
(function () {
  try { console.warn('[guided-debug] guided_builder_v2.broken.js disabled (archived)'); } catch (e) {}
})();
        //     select(b, 'MA type', 'ma_type', [['ema', 'EMA'], ['sma', 'SMA']], (rule && rule.ma_type != null) ? rule.ma_type : 'ema');
        //     inputNumber(b, 'MA length', 'length', (rule && rule.length != null) ? rule.length : 20, { min: 1 });
        //     inputNumber(b, 'Min % above MA', 'min_pct', (rule && rule.min_pct != null) ? rule.min_pct : 0.1, { step: 0.01, min: 0 });
        //     b.appendChild(el('div', { class: 'muted', html: 'True if volume is at least min % above the MA.' }));
        //   });
        // }

        // Initialize row UI (set selected type, show param block, wire inputs)
        (function initRow() {
          try {
            // prefer explicitly provided rule type, otherwise use the first option
            const initialType = (rule && rule.type) ? String(rule.type) : (typeSel.options && typeSel.options.length ? typeSel.options[0].value : '');
            if (initialType) typeSel.value = initialType;
          } catch (e) {}

          try {
            if ((section === 'signal' || section === 'trigger') && tfSel) {
              tfSel.value = (rule && rule.tf) ? String(rule.tf) : (tfSel.querySelector('option[value="default"]') ? 'default' : (tfSel.options[0] ? tfSel.options[0].value : ''));
            }
          } catch (e) {}

          function updateTypeVisibility() {
            const sel = String(typeSel.value || '');
            for (const b of fields.querySelectorAll('[data-type]')) {
              b.style.display = (b.getAttribute('data-type') === sel) ? '' : 'none';
            }
          }

          typeSel.addEventListener('change', () => {
            updateTypeVisibility();
            try { syncHidden(); } catch (e) {}
            try { schedule(); } catch (e) {}
          });

          // wire inner controls to update preview/state
          for (const inp of fields.querySelectorAll('[data-field]')) {
            inp.addEventListener('input', () => { try { syncHidden(); schedule(); } catch (e) {} });
            inp.addEventListener('change', () => { try { syncHidden(); schedule(); } catch (e) {} });
          }

          updateTypeVisibility();
        })();

        return row;

      const serialize = (container) => {
        const rules = [];
        if (!container) return rules;
        const rows = container.querySelectorAll('.rule-row');
        for (const row of rows) {
          const rule = {};
          const typeEl = row.querySelector('[data-field="type"]');
          if (!typeEl) continue;
          rule.type = String(typeEl.value || '').trim();

          const tfEl = row.querySelector('[data-field="tf"]');
          if (tfEl && String(tfEl.value || '') !== 'default') rule.tf = String(tfEl.value || '');

          const validEl = row.querySelector('[data-field="valid_for_bars"]');
          if (validEl) {
            const v = String(validEl.value || '').trim();
            const n = Number(v);
            if (v !== '' && Number.isFinite(n)) rule.valid_for_bars = n;
          }

          const block = row.querySelector(`[data-type="${rule.type}"]`);
          if (block) {
            const inputs = block.querySelectorAll('[data-field]');
            for (const inp of inputs) {
              const key = inp.getAttribute('data-field');
              if (!key) continue;
              const raw = inp.value != null ? String(inp.value).trim() : '';
              if (raw === '') continue;
              const num = Number(raw);
              rule[key] = (Number.isFinite(num) && raw === String(num)) ? num : raw;
            }
          }

          rules.push(rule);
        }
        return rules;
      }

      function syncHidden() {
        if (ctxHidden) ctxHidden.value = JSON.stringify(serialize(ctxContainer));
        if (sigHidden) sigHidden.value = JSON.stringify(serialize(sigContainer));
        if (trgHidden) trgHidden.value = JSON.stringify(serialize(trgContainer));
      }

      function loadInitial() {
        const ctx = safeParseJson(ctxHidden?.value);
        const sig = safeParseJson(sigHidden?.value);
        const trg = safeParseJson(trgHidden?.value);

        if (ctxContainer) {
          ctxContainer.innerHTML = '';
          for (const r of ctx) ctxContainer.appendChild(ruleRow('context', r));
        }
        if (sigContainer) {
          sigContainer.innerHTML = '';
          for (const r of sig) sigContainer.appendChild(ruleRow('signal', r));
        }
        if (trgContainer) {
          trgContainer.innerHTML = '';
          for (const r of trg) trgContainer.appendChild(ruleRow('trigger', r));
        }
        syncHidden();
      }

      function defaultPrimaryContextRule(type) {
        const t = String(type || '').trim();
        if (t === 'price_vs_ma') return { type: 'price_vs_ma', ma_type: 'ema', length: 200 };
        if (t === 'ma_cross_state') return { type: 'ma_cross_state', ma_type: 'ema', fast: 20, slow: 50 };
        if (t === 'atr_pct') return { type: 'atr_pct', atr_len: 14, op: '>', threshold: 0.005 };
        if (t === 'structure_breakout_state') return { type: 'structure_breakout_state', length: 20 };
        if (t === 'ma_spread_pct') return { type: 'ma_spread_pct', ma_type: 'ema', fast: 20, slow: 200, threshold: 0.01 };
        if (t === 'custom') return { type: 'custom', bull_expr: '', bear_expr: '' };
        return { type: t || 'price_vs_ma' };
      }

      function applyPrimaryContextType() {
        const t = String(primaryContextSel?.value || 'ma_stack');
        if (maStackBlock) {
          maStackBlock.style.display = (t === 'ma_stack') ? '' : 'none';
        }

        // For non-MA-stack primary contexts, seed (or retarget) the first context rule.
        if (t !== 'ma_stack' && t !== 'none') {
          if (ctxContainer) {
            const firstRow = ctxContainer.querySelector('.rule-row');
            if (!firstRow) {
              ctxContainer.prepend(ruleRow('context', defaultPrimaryContextRule(t)));
            } else {
              const typeEl = firstRow.querySelector('[data-field="type"]');
              if (typeEl && typeEl.value !== t) {
                typeEl.value = t;
                typeEl.dispatchEvent(new Event('change', { bubbles: true }));
              }
            }
          }
        }

        syncHidden();
      }

      const addContextBtn = document.getElementById('add-context');
      if (addContextBtn) {
        addContextBtn.addEventListener('click', () => {
          if (ctxContainer) ctxContainer.appendChild(ruleRow('context', { type: 'price_vs_ma' }));
          syncHidden();
          schedule();
        });
      }
      const addSignalBtn = document.getElementById('add-signal');
      if (addSignalBtn) {
        addSignalBtn.addEventListener('click', () => {
          if (sigContainer) sigContainer.appendChild(ruleRow('signal', { type: 'rsi_threshold' }));
          syncHidden();
          schedule();
        });
      }
      const addTriggerBtn = document.getElementById('add-trigger');
      if (addTriggerBtn) {
        addTriggerBtn.addEventListener('click', () => {
          if (trgContainer) trgContainer.appendChild(ruleRow('trigger', { type: 'prior_bar_break' }));
          syncHidden();
          schedule();
        });
      }

      function updateTriggerParamVisibility() {
        const t = String(val('trigger_type') || 'pin_bar');
        const show = (id, on) => {
          const el = document.getElementById(id);
          if (!el) return;
          el.style.display = on ? '' : 'none';
        };
        show('trigger_pin_bar', t === 'pin_bar');
        show('trigger_ma_reclaim', t === 'ma_reclaim');
        show('trigger_donchian', t === 'donchian_breakout');
        show('trigger_range', t === 'range_breakout');
        show('trigger_wide_range', t === 'wide_range_candle');
        show('trigger_custom', t === 'custom');
      }

      let ctxVisTimer = null;
      async function updateContextVisual(payload, opts) {
        if (!contextVisualEl || !contextVisualMsgEl) return;
        if (!window.Plotly) {
          contextVisualMsgEl.textContent = 'Plotly is not available; context preview disabled.';
          return;
        }

        const force = !!(opts && opts.force);

        // Debounce to avoid spamming the server on every keystroke.
        if (ctxVisTimer) window.clearTimeout(ctxVisTimer);
        const delayMs = force ? 0 : 350;
        ctxVisTimer = window.setTimeout(async () => {
          try {
            contextVisualMsgEl.textContent = 'Loading context preview…';
            const res = await fetch('/api/guided/builder_v2/context_visual', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              cache: 'no-store',
              body: JSON.stringify(payload)
            });
            const out = await res.json();
            if (!out || !out.ok) {
              contextVisualMsgEl.textContent = (out && out.message) ? String(out.message) : 'Context preview unavailable.';
              if (contextVisualEl) contextVisualEl.innerHTML = '';
              return;
            }
            contextVisualMsgEl.textContent = out.message ? String(out.message) : '';
            if (out.fig && out.fig.data && out.fig.layout) {
              await window.Plotly.react(contextVisualEl, out.fig.data, out.fig.layout, {
                displayModeBar: false,
                responsive: true,
              });
            }
          } catch (e) {
            contextVisualMsgEl.textContent = 'Context preview unavailable.';
            if (contextVisualEl) contextVisualEl.innerHTML = '';
          }
        }, delayMs);
      }

      let setupVisTimer = null;
      async function updateSetupVisual(payload, opts) {
        if (!setupVisualEl || !setupVisualMsgEl) return;
        if (!window.Plotly) {
          setupVisualMsgEl.textContent = 'Plotly is not available; setup preview disabled.';
          return;
        }

        const force = !!(opts && opts.force);

        if (setupVisTimer) window.clearTimeout(setupVisTimer);
        const delayMs = force ? 0 : 450;
        setupVisTimer = window.setTimeout(async () => {
          try {
            setupVisualMsgEl.textContent = 'Loading setup preview…';
            const res = await fetch('/api/guided/builder_v2/setup_visual', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              cache: 'no-store',
              body: JSON.stringify(payload)
            });
            const out = await res.json();
            if (!out || !out.ok) {
              setupVisualMsgEl.textContent = (out && out.message) ? String(out.message) : 'Setup preview unavailable.';
              if (setupVisualEl) setupVisualEl.innerHTML = '';
              return;
            }
            setupVisualMsgEl.textContent = out.message ? String(out.message) : '';
            if (out.fig && out.fig.data && out.fig.layout) {
              await window.Plotly.react(setupVisualEl, out.fig.data, out.fig.layout, {
                displayModeBar: false,
                responsive: true,
              });
            }
          } catch (e) {
            setupVisualMsgEl.textContent = 'Setup preview unavailable.';
            if (setupVisualEl) setupVisualEl.innerHTML = '';
          }
        }, delayMs);
      }

      function buildPayload() {
        syncHidden();
        updateTriggerParamVisibility();
        applyPrimaryContextType();
        return {
          entry_tf: val('entry_tf') || '1h',
          context_tf: val('context_tf') || '',
          signal_tf: val('signal_tf') || '',
          trigger_tf: val('trigger_tf') || '',
          primary_context_type: val('primary_context_type') || 'ma_stack',
          align_with_context: !!val('align_with_context'),
          long_enabled: !!val('long_enabled'),
          short_enabled: !!val('short_enabled'),
          ma_type: val('ma_type') || 'ema',
          ma_fast: num('ma_fast'),
          ma_mid: num('ma_mid'),
          ma_slow: num('ma_slow'),
          stack_mode: val('stack_mode') || 'none',
          slope_mode: val('slope_mode') || 'none',
          slope_lookback: num('slope_lookback'),
          min_ma_dist_pct: num('min_ma_dist_pct'),
          trigger_type: val('trigger_type') || 'pin_bar',
          trigger_valid_for_bars: num('trigger_valid_for_bars'),
          pin_wick_body: num('pin_wick_body'),
          pin_opp_wick_body_max: num('pin_opp_wick_body_max'),
          pin_min_body_pct: num('pin_min_body_pct'),
          trigger_ma_type: val('trigger_ma_type') || 'ema',
          trigger_ma_len: num('trigger_ma_len'),
          trigger_don_len: num('trigger_don_len'),
          trigger_range_len: num('trigger_range_len'),
          trigger_atr_len: num('trigger_atr_len'),
          trigger_atr_mult: num('trigger_atr_mult'),
          trigger_custom_bull_expr: val('trigger_custom_bull_expr') || '',
          trigger_custom_bear_expr: val('trigger_custom_bear_expr') || '',
          context_rules: safeParseJson(ctxHidden?.value),
          signal_rules: safeParseJson(sigHidden?.value),
          trigger_rules: safeParseJson(trgHidden?.value)
        };
      }

      async function updatePreview() {
        const payload = buildPayload();

        // Update the context chart in parallel with the textual preview.
        updateContextVisual(payload);
        updateSetupVisual(payload);

        try {
          const res = await fetch('/api/guided/builder_v2/preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            cache: 'no-store',
            body: JSON.stringify(payload)
          });
          if (!res.ok) throw new Error('preview request failed');
          const preview = await res.json();

          let html = `<div class="muted" style="margin-bottom:10px;">Context TF: <strong>${escapeHtml(preview.context_tf)}</strong>. Entry TF: <strong>${escapeHtml(preview.entry_tf)}</strong>. Signals TF: <strong>${escapeHtml(preview.signal_tf)}</strong>. Triggers TF: <strong>${escapeHtml(preview.trigger_tf)}</strong>.</div>`;
          if (preview.aligned_dual) {
            html += `<div class="muted" style="margin-bottom:10px;">Aligned mode: LONG=bull, SHORT=bear</div>`;
          }
          html += renderSide('LONG', preview.long);
          html += renderSide('SHORT', preview.short);
          body.innerHTML = html;
        } catch (e) {
          body.innerHTML = `<div class="muted">Preview unavailable.</div>`;
        }
      }

      let t = null;
      function schedule() {
        if (t) window.clearTimeout(t);
        t = window.setTimeout(updatePreview, 150);
      }

      form.addEventListener('input', schedule);
      form.addEventListener('change', schedule);
      if (primaryContextSel) {
        primaryContextSel.addEventListener('change', () => {
          applyPrimaryContextType();
          schedule();
        });
      }
      form.addEventListener('submit', () => {
        syncHidden();
      });

      const reloadCtxBtn = document.getElementById('reload-context-visual');
      if (reloadCtxBtn) {
        reloadCtxBtn.addEventListener('click', () => {
          const payload = buildPayload();
          updateContextVisual(payload, { force: true });
        });
      }

      const reloadSetupBtn = document.getElementById('reload-setup-visual');
      if (reloadSetupBtn) {
        reloadSetupBtn.addEventListener('click', () => {
          const payload = buildPayload();
          updateSetupVisual(payload, { force: true });
        });
      }

      // If the user navigates back to this page, some browsers restore from bfcache.
      // Force-refresh the previews/charts on pageshow.
      window.addEventListener('pageshow', () => {
        updatePreview();
      });

      loadInitial();
      applyPrimaryContextType();
      updatePreview();
    })();
  