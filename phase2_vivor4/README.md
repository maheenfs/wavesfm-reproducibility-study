Phase 2 WavesFM Benchmark Workspace

Purpose
- Reproduce WavesFM downstream benchmark results in isolated local sessions.
- Keep official references, local runs, plots, summaries, and comparisons clearly separated.
- Make it easy to pull only the files needed for review or Mac-side comparison.

Canonical workflow
1. Build or transfer caches into `datasets_h5/`.
2. Validate the target session:
   `python3 phase2_vivor4/scripts/preflight_check.py --session-root phase2_vivor4/runs/<session>`
3. Launch the managed pipeline:
   `python3 phase2_vivor4/scripts/wait_for_radcom_and_run_next.py --session-root phase2_vivor4/runs/<session>`
4. Read the live dashboard at `phase2_vivor4/automation_logs/dashboard.html`.
5. Use the session manifests and per-task bundles to pull or compare artifacts.

Second-pass rerun rule
- Do not send a rerun into the top-level `phase2_vivor4/local_results/` tree.
- For reruns that must remain report-distinct from a previous attempt, always create a fresh named session under `phase2_vivor4/runs/<session>/`.
- For the current `sensing` / `deepmimo-los` / `deepmimo-beam` second pass, use:
  `bash phase2_vivor4/scripts/prepare_second_try_sensing_deepmimo.sh`
- That helper archives the current generated phase2 state, prepares only the three second-pass tasks, and stamps a dedicated session name beginning with `second_try_sensing_deepmimo_`.

Experimental-study rule
- Reduced-data or subset fine-tuning studies are not part of the main
  reproduction tree.
- Store them in their own fresh session roots under `phase2_vivor4/runs/<session>/`.
- For the modulation subset follow-up study, see
  `notes/05_modulation_subset_experiments.md`.
- To scaffold the modulation subset study tables, launch scripts, and planned
  session roots before running it, use:
  `python3 phase2_vivor4/scripts/prepare_modulation_subset_study.py`

Session layout
- `runs/<session>/local_results/by_task/<task>/<mode>/s<seed>/`: raw run outputs.
- `runs/<session>/local_results/summaries/summary_manifest.json`: local summary index.
- `runs/<session>/local_results/summaries/by_task/<task>/<mode>.json`: per-task-mode summary bundle.
- `runs/<session>/comparisons/comparison_manifest.json`: local-vs-official comparison index.
- `runs/<session>/comparisons/by_task/<task>/<mode>.json`: per-task-mode comparison bundle.
- `runs/<session>/plots/plot_manifest.json`: generated plot manifest.
- `runs/<session>/session_manifest.json`: session environment and command manifest.
- `runs/<session>/preflight_report.json`: preflight validation report.

Reference data
- `official_results/by_task/<task>.json`: official task references.
- `official_results/official_results_all.json`: full official reference export.

Notes
- The canonical codebase is `wavesfm/`, not the duplicated top-level training tree.
- `run_all_tasks.py` is still available for direct launches, but the managed tracker is the recommended path because it keeps all output paths synchronized.
- `prepare_clean_launch_root.py` now emits this same session-based preflight + tracker workflow when preparing a clean remote launch.
- `prepare_clean_launch_root.py` now also accepts:
  - `--tasks ...` to restrict a prepared session to a task subset
  - `--session-name ...` to stamp a report-friendly rerun/session label without colliding with an older session directory
- For long unattended remote runs, prefer the helper-generated detached launcher under `phase2_vivor4/runs/`; it uses `tmux` when available and falls back to `nohup`.
- The current harness default is `--num-workers 4` on `ngwn06`, because the project-venv preflight worker smoke test passed there for `1` and `4`.
- Keep using `preflight_check.py --num-workers <n>` as the gate on any other host, and fall back to `--num-workers 0` if worker startup or long-run stability becomes a problem.
- The dashboard now includes browser-side SSH-tunnel / dashboard-link monitoring:
  - it records the last successful browser fetch of `after_radcom_status.json`
  - marks the link as `connected`, `stale`, `reconnecting`, or `lost`
  - shows time since the last success and the most recent outage reason/duration
  - keeps retrying automatically and reloads the page after a successful reconnection when auto refresh is enabled
  - preserves the last known remote training snapshot while disconnected
  - exposes resilient Mac tunnel commands (`while true; do ssh ...; done`, `autossh -M 0 ...`, and `dashboard_tunnel_watch.sh`) in the dashboard so the tunnel itself can auto-reconnect outside the browser
  - the browser can monitor and retry the dashboard path, but it cannot recreate the SSH tunnel by itself
  - repeated identical browser-visible failures now trigger longer cooldown retries instead of probing every few seconds forever
  - the dashboard also includes `Process Liveness`, `Disk And Checkpoint Watch`, `Checkpoint Timeline`, `GPU And Host Trends`, and `GPU Safety Guard` cards for remote monitoring
  - the GPU safety guard pauses the active training queue after sustained hot temperature readings, resumes automatically after the cooldown window and repeated cool readings, and stops the queue if the GPU stays in a critical range
  - every GPU-guard pause, resume, and stop action is recorded with timestamps, reason text, and risk-reduction guidance
- Comparison output now defaults to the local session scope only, so partial runs do not produce unrelated `missing_local_result` rows unless explicitly requested.
