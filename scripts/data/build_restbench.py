"""Reproducible builder for the 150-instance RestBench benchmark.

RestBench (RestGPT, Song et al. 2023) ships two real instruction sets:
  - datasets/tmdb.json    : 100 queries over the TMDB REST API
  - datasets/spotify.json :  57 queries over the Spotify Web API
each item being {"query": str, "solution": [ordered gold endpoints]}.

We emit 150 instances in the repo's unified schema:
  {id, query, available_tools:[{name,description,parameters}],
   gold_plan:[{tool, args}], gold_answer, mock_outputs, metric}

available_tools is the FULL endpoint universe for the relevant API so the
selector faces real distractors:
  - TMDB    : the 55-endpoint universe vendored at scripts/data/tmdb_tools.json
              -- the authoritative RestBench TMDB endpoint list.
  - Spotify : the 40-endpoint universe parsed straight from RestGPT's
              specs/spotify_oas.json (the OpenAPI spec RestBench ships).

gold_plan is the gold API path as [{tool: "GET /endpoint", args: {}}], matching
the official RestBench "Correct Path" metric (src/evaluators.py): the gold
endpoint sequence must be an ordered subsequence of the successfully-executed
endpoints, and the offline executor marks any tool whose name is in
available_tools as a success. So every gold endpoint MUST appear by name in
available_tools.

Composition: all 100 TMDB queries + the first 50 Spotify queries = 150.

Data-cleaning (real source has two artifacts, fixed deterministically):
  - some TMDB solution endpoints carry stray leading/trailing whitespace
    (" GET /movie/popular", "GET /search/movie ") -> stripped.
  - one Spotify solution uses the singular "GET /track/{id}"; the authoritative
    Spotify OAS (and the rest of the dataset) uses the plural "GET /tracks/{id}"
    -> normalized to the OAS endpoint name.

Run:
  python scripts/data/build_restbench.py
"""

from __future__ import annotations

import json
import os
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "data", "restbench", "samples.json")
TMDB_UNIVERSE_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmdb_tools.json")

RAW = "https://raw.githubusercontent.com/Yifan-Song793/RestGPT/main"
SOURCES = {
    "tmdb_queries": f"{RAW}/datasets/tmdb.json",
    "spotify_queries": f"{RAW}/datasets/spotify.json",
    "spotify_oas": f"{RAW}/specs/spotify_oas.json",
}
# /tmp cache so the build is reproducible even when the home quota is full.
CACHE_DIR = "/tmp/restgpt_src_cache"

N_TMDB = 100
N_SPOTIFY = 50

# Endpoint-name normalization: the singular /track/{id} is a typo for the
# Spotify OAS endpoint /tracks/{id}.
ENDPOINT_FIXES = {"GET /track/{id}": "GET /tracks/{id}"}


def _fetch(name: str) -> dict | list:
    """Return parsed JSON for a source, downloading once and caching to /tmp."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, name + ".json")
    if os.path.exists(cache) and os.path.getsize(cache) > 0:
        try:
            with open(cache) as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass  # corrupt cache -> re-download
    url = SOURCES[name]
    with urllib.request.urlopen(url, timeout=90) as r:
        data = r.read()
    with open(cache, "wb") as f:
        f.write(data)
    return json.loads(data)


def _clean_endpoint(ep: str) -> str:
    ep = ep.strip()
    return ENDPOINT_FIXES.get(ep, ep)


# --------------------------------------------------------------------------- #
# Tool universes
# --------------------------------------------------------------------------- #
def load_tmdb_universe() -> list[dict]:
    """Reuse the authoritative TMDB endpoint universe vendored with the repo."""
    with open(TMDB_UNIVERSE_SRC) as f:
        tools = json.load(f)
    # Drop any synthetic non-endpoint tools (e.g. a 'verify' crutch) if present.
    tools = [t for t in tools if " " in t["name"] and t["name"].split(" ", 1)[0].isupper()]
    # de-dupe by name, preserve order
    seen, out = set(), []
    for t in tools:
        if t["name"] not in seen:
            seen.add(t["name"])
            out.append(t)
    return out


def _resolve_params(op: dict, oas: dict) -> dict:
    """Flatten an OpenAPI operation's parameters into {name: '(in) description'}."""
    comp_params = oas.get("components", {}).get("parameters", {})
    params: dict[str, str] = {}
    for par in op.get("parameters", []) or []:
        if "$ref" in par:
            ref = par["$ref"].split("/")[-1]
            par = comp_params.get(ref, {})
        name = par.get("name")
        if not name:
            continue
        loc = par.get("in", "query")
        desc = (par.get("schema", {}) or {}).get("description") or par.get("description") or ""
        desc = " ".join(desc.split())[:160]
        params[name] = f"({loc}) {desc}".rstrip()
    # request-body backed write ops get a generic body param so the schema is non-empty
    if op.get("requestBody") and not params:
        params["body"] = "(body) request payload"
    return params


def load_spotify_universe() -> list[dict]:
    """Parse the full 40-endpoint Spotify universe from RestGPT's OpenAPI spec."""
    oas = _fetch("spotify_oas")
    tools = []
    for path, methods in oas.get("paths", {}).items():
        for method, op in methods.items():
            if method.lower() not in ("get", "post", "put", "delete", "patch"):
                continue
            name = f"{method.upper()} {path}"
            summary = (op.get("summary") or "").strip()
            description = " ".join((op.get("description") or "").split())[:240]
            full_desc = ". ".join([s for s in (summary, description) if s]) or name
            tools.append({
                "name": name,
                "description": full_desc,
                "parameters": _resolve_params(op, oas),
            })
    tools.sort(key=lambda t: t["name"])
    return tools


# --------------------------------------------------------------------------- #
# Instances
# --------------------------------------------------------------------------- #
def build_instances(queries: list[dict], universe: list[dict], prefix: str,
                    limit: int) -> list[dict]:
    names = {t["name"] for t in universe}
    out = []
    for i, item in enumerate(queries[:limit], start=1):
        plan = [{"tool": _clean_endpoint(ep), "args": {}} for ep in item["solution"]]
        for step in plan:
            if step["tool"] not in names:
                raise ValueError(
                    f"{prefix}_{i}: gold endpoint {step['tool']!r} absent from the "
                    f"{prefix} tool universe -- gold would fail the official metric."
                )
        out.append({
            "id": f"{prefix}_{i}",
            "query": item["query"],
            "available_tools": universe,
            "gold_plan": plan,
            "gold_answer": None,
            "mock_outputs": {},
            "metric": "restbench_correct_path",
        })
    return out


def main() -> None:
    tmdb_queries = _fetch("tmdb_queries")
    spotify_queries = _fetch("spotify_queries")

    tmdb_universe = load_tmdb_universe()
    spotify_universe = load_spotify_universe()
    print(f"TMDB universe   : {len(tmdb_universe)} endpoints")
    print(f"Spotify universe: {len(spotify_universe)} endpoints")

    tmdb = build_instances(tmdb_queries, tmdb_universe, "rb_tmdb", N_TMDB)
    spotify = build_instances(spotify_queries, spotify_universe, "rb_spotify", N_SPOTIFY)
    instances = tmdb + spotify
    assert len(instances) == 150, f"expected 150, got {len(instances)}"

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(instances, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(instances)} instances -> {OUT} "
          f"({len(tmdb)} TMDB + {len(spotify)} Spotify)")


if __name__ == "__main__":
    main()
