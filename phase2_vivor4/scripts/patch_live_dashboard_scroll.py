#!/usr/bin/env python3
"""Inject scroll preservation into a live generated dashboard.

The tracker process renders dashboard.html from its in-memory Python source.
When the running tracker predates a dashboard JavaScript fix, this patcher keeps
the served HTML usable until the tracker is restarted with the updated source.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


MARKER = "dashboard-scroll-preserve-hotfix"

HOTFIX = f"""
<script id="{MARKER}">
(function () {{
  const key = "wavesfm_dashboard_scroll_hotfix_v1:" + window.location.pathname;

  function saveScrollPosition() {{
    try {{
      window.localStorage.setItem(key, JSON.stringify({{
        x: window.scrollX || 0,
        y: window.scrollY || 0,
        ts: Date.now(),
      }}));
    }} catch (error) {{}}
  }}

  function restoreScrollPosition() {{
    try {{
      if ("scrollRestoration" in window.history) {{
        window.history.scrollRestoration = "manual";
      }}
      const raw = window.localStorage.getItem(key);
      if (!raw) return;
      const saved = JSON.parse(raw);
      const ageMs = Date.now() - Number(saved.ts || 0);
      if (!Number.isFinite(ageMs) || ageMs > 60 * 60 * 1000) return;
      const targetX = Math.max(0, Number(saved.x || 0));
      const targetY = Math.max(0, Number(saved.y || 0));
      window.requestAnimationFrame(() => {{
        window.requestAnimationFrame(() => {{
          const maxY = Math.max(0, document.documentElement.scrollHeight - window.innerHeight);
          window.scrollTo({{ left: targetX, top: Math.min(targetY, maxY), behavior: "auto" }});
        }});
      }});
    }} catch (error) {{}}
  }}

  const originalForceHardReload = window.forceHardReload;
  if (typeof originalForceHardReload === "function" && !originalForceHardReload.__scrollPreserveHotfix) {{
    const wrapped = function (...args) {{
      saveScrollPosition();
      return originalForceHardReload.apply(this, args);
    }};
    wrapped.__scrollPreserveHotfix = true;
    window.forceHardReload = wrapped;
  }}

  window.addEventListener("beforeunload", saveScrollPosition);
  window.addEventListener("pagehide", saveScrollPosition);
  restoreScrollPosition();
}})();
</script>
""".strip()


def patch_dashboard(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return False
    marker = "</body>"
    idx = text.rfind(marker)
    if idx < 0:
        return False
    patched = text[:idx] + HOTFIX + "\n" + text[idx:]
    tmp = path.with_suffix(path.suffix + ".scrollpatch.tmp")
    tmp.write_text(patched, encoding="utf-8")
    tmp.replace(path)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dashboard",
        nargs="?",
        default="phase2_vivor4/automation_logs/dashboard.html",
        type=Path,
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dashboard = args.dashboard
    if not args.watch:
        changed = patch_dashboard(dashboard)
        print(f"{dashboard}: {'patched' if changed else 'unchanged'}")
        return

    while True:
        changed = patch_dashboard(dashboard)
        if changed:
            print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} patched {dashboard}", flush=True)
        time.sleep(max(args.interval, 0.2))


if __name__ == "__main__":
    main()
