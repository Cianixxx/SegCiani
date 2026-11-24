"""List the Gemini models available with your API key.

Model availability changes over time (e.g. gemini-2.0-flash was retired). Run
this to see which models you can actually use, then pass one via --model::

    python scripts/list_gemini_models.py

Needs GOOGLE_API_KEY (or GEMINI_API_KEY) set in the environment.
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    try:
        from google import genai
    except ImportError:
        print("Install first: python -m pip install google-genai", file=sys.stderr)
        sys.exit(1)

    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        print("Set GOOGLE_API_KEY first (PowerShell: $env:GOOGLE_API_KEY=\"...\").",
              file=sys.stderr)
        sys.exit(1)

    client = genai.Client()
    print("Models that support generateContent:\n")
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if not actions or "generateContent" in actions:
            name = m.name.replace("models/", "")
            disp = getattr(m, "display_name", "") or ""
            print(f"  {name:35s} {disp}")
    print("\nUse one with: --model <name>  (e.g. gemini-2.5-flash, gemini-2.5-pro)")


if __name__ == "__main__":
    main()
