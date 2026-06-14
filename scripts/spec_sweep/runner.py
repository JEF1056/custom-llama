"""Staged-greedy sweep orchestrator with ETA and crash-resume.

Stages (each holds prior winners fixed; cheap 25k stages first):
  A1  spec-type           — draft-mtp,ngram-mod | ngram-mod,draft-mtp | mtp
  A2  spec-draft-p-min     — 0.1 | 0.2 | 0.3            (only if winner has draft-mtp)
  A3  ngram-mod params     — n-max {8,16,32}, n-match {4,8}  (only if winner has ngram)
  B   160k validation      — Stage-A winner vs baseline
  C   ctx / parallel       — single196 | maxctx256 | parallel2  (text+code @ 90k)

Resumable: every completed config and every stage decision is persisted to
state.json; results stream to results.csv. Re-running skips finished work.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from . import config as C
from . import harness, payloads

# ── ETA weighting (relative cost; auto-calibrates to seconds after run 1) ──────
CTX_WEIGHT = {"25k": 1.0, "90k": 3.5, "160k": 6.5}
RESTART_WEIGHT = 0.5


def _reps(ctx: str) -> int:
    return C.REPEATS.get(ctx, C.REPEATS_DEFAULT)


def _measurement_weight(measurements) -> float:
    return RESTART_WEIGHT + sum(CTX_WEIGHT[ctx] * _reps(ctx) for ctx, *_ in measurements)


# Upper-bound provisional plan weight (all conditional stages fire).
def _provisional_total() -> float:
    r25, r90, r160 = _reps("25k"), _reps("90k"), _reps("160k")
    mid2 = RESTART_WEIGHT + 2 * CTX_WEIGHT["25k"] * r25
    long2 = RESTART_WEIGHT + 2 * CTX_WEIGHT["160k"] * r160
    slot2 = RESTART_WEIGHT + 2 * CTX_WEIGHT["90k"] * r90
    # A1=3, A2=2, A3=3, A4=2 (mid) ; B=3 (long) ; C=3 (slot)
    return 10 * mid2 + 3 * long2 + 3 * slot2


# ── state persistence ─────────────────────────────────────────────────────────
def _load_state() -> dict:
    if C.STATE_JSON.exists():
        return json.loads(C.STATE_JSON.read_text())
    return {"completed": {}, "decisions": {}, "weight_done": 0.0, "time_done": 0.0}


def _save_state(state: dict) -> None:
    C.STATE_JSON.write_text(json.dumps(state, indent=2))


# ── results.csv ───────────────────────────────────────────────────────────────
def _log_rows(tag: str, cfg: dict, rows: list[dict]) -> None:
    new = not C.RESULTS_CSV.exists()
    with C.RESULTS_CSV.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["tag", "ctx", "workload", "tg", "tg_cv", "tg_runs",
                        "tg_per_stream", "ctok", "ptok", "ttft"] + C.TRACKED_KEYS)
        for r in rows:
            w.writerow([
                tag, r["ctx"], r["workload"], r["tg"], r.get("tg_cv", 0.0),
                "|".join(map(str, r.get("tg_runs", []))),
                "|".join(map(str, r.get("tg_per_stream", []))),
                r["ctok"], r["ptok"], r["ttft"],
            ] + [str(cfg.get(k, "")).replace(",", "/") for k in C.TRACKED_KEYS])


def _score(tag: str, ctx: str):
    """mean(text_tg, code_tg) for tag at ctx; last row wins per workload."""
    if not C.RESULTS_CSV.exists():
        return None
    wl: dict[str, float] = {}
    for row in csv.DictReader(C.RESULTS_CSV.open()):
        if row["tag"] == tag and row["ctx"] == ctx:
            wl[row["workload"]] = float(row["tg"])
    if "text" in wl and "code" in wl:
        return (wl["text"] + wl["code"]) / 2, wl
    return None


def _ttft(tag: str, ctx: str) -> dict:
    """Per-workload cold TTFT (seconds) for tag at ctx; last row wins."""
    out: dict[str, float] = {}
    if not C.RESULTS_CSV.exists():
        return out
    for row in csv.DictReader(C.RESULTS_CSV.open()):
        if row["tag"] == tag and row["ctx"] == ctx:
            try:
                out[row["workload"]] = float(row["ttft"])
            except (KeyError, ValueError):
                pass
    return out


def _fmt_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class Sweep:
    def __init__(self):
        self.state = _load_state()
        self.total_weight = _provisional_total()

    # ── run one config (restart + measurements), with resume + ETA ────────────
    def run_config(self, tag: str, params: dict, measurements: list) -> None:
        weight = _measurement_weight(measurements)
        if tag in self.state["completed"]:
            print(f"[skip] {tag} (already done)", flush=True)
            return

        cfg_desc = " ".join(f"{k}={v}" for k, v in params.items())
        print(f"\n=== {tag} :: {cfg_desc}", flush=True)
        t0 = time.time()
        harness.set_params(params)
        harness.restart()
        try:
            waited = harness.wait_ready()
            print(f"    model ready in {waited:.0f}s; measuring…", flush=True)
            snapshot = harness.read_params(C.TRACKED_KEYS)
            rows = []
            for ctx, workload, payload_name, concurrency in measurements:
                res = harness.measure(C.PAYLOAD_DIR / f"{payload_name}.json",
                                      concurrency, repeats=_reps(ctx))
                res.update({"ctx": ctx, "workload": workload})
                rows.append(res)
                extra = f" per-stream={res['tg_per_stream']}" if concurrency > 1 else ""
                print(f"    {ctx:>4} {workload:<4} tg={res['tg']:>6} t/s "
                      f"(±{res['tg_cv']:.1f}% n={len(res['tg_runs'])}) "
                      f"ttft={res['ttft']}s ptok={res['ptok']}{extra}", flush=True)
        except (RuntimeError, OSError, ValueError) as e:
            # Non-viable config (crash on load, unsupported flag, network drop):
            # record tg=0 so it loses ranking, then continue the sweep.
            print(f"    !! {tag} non-viable: {e} -- recording tg=0", flush=True)
            snapshot = harness.read_params(C.TRACKED_KEYS)
            rows = [{"ctx": ctx, "workload": workload, "tg": 0.0,
                     "tg_cv": 0.0, "tg_runs": [], "tg_per_stream": [],
                     "ctok": 0, "ptok": 0, "ttft": 0.0}
                    for ctx, workload, *_ in measurements]
        _log_rows(tag, snapshot, rows)

        elapsed = time.time() - t0
        self.state["completed"][tag] = {"weight": weight, "elapsed": round(elapsed, 1)}
        self.state["weight_done"] += weight
        self.state["time_done"] += elapsed
        _save_state(self.state)

        rate = self.state["time_done"] / max(self.state["weight_done"], 1e-6)
        remaining = max(self.total_weight - self.state["weight_done"], 0)
        print(f"    config done in {elapsed:.0f}s | "
              f"ETA remaining ~{_fmt_eta(rate * remaining)}", flush=True)

    # ── stage helpers ─────────────────────────────────────────────────────────
    def _best(self, tags, ctx="25k"):
        scored = [(t, self._safe_score(t, ctx)) for t in tags]
        scored = [(t, s) for t, s in scored if s]
        scored.sort(key=lambda x: x[1][0], reverse=True)
        return scored

    @staticmethod
    def _safe_score(tag, ctx):
        return _score(tag, ctx)

    def _cfg(self, **over):
        d = dict(C.BASELINE)
        d.update(over)
        return d

    # ── the stages ────────────────────────────────────────────────────────────
    def run(self):
        payloads.ensure()
        C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        if not C.INI_BACKUP.exists():
            C.INI_BACKUP.write_text(C.INI_PATH.read_text())
            print(f"[backup] models.ini -> {C.INI_BACKUP}", flush=True)

        dec = self.state["decisions"]
        MID = [("25k", "text", "text_25k", 1), ("25k", "code", "code_25k", 1)]
        LONG = [("160k", "text", "text_160k", 1), ("160k", "code", "code_160k", 1)]

        # ---- Stage A1: spec-type ----
        # NOTE: ``ngram-mod`` alone is non-viable on this fork — the ngram
        # speculative path still asserts a draft context (ctx_dft), which only
        # exists when a draft model (draft-mtp) is loaded. So every candidate
        # here includes draft-mtp; we test mtp-only and the two hybrid orders.
        print("\n########## STAGE A1: spec-type ##########", flush=True)
        A1 = {
            "A1_mtp_ngram": "draft-mtp,ngram-mod",
            "A1_ngram_mtp": "ngram-mod,draft-mtp",
            "A1_mtp_only": "draft-mtp",
        }
        for tag, st in A1.items():
            self.run_config(tag, self._cfg(**{"spec-type": st}), MID)
        rank = self._best(list(A1))
        print("A1 ranking:", [(t, round(s[0], 2)) for t, s in rank], flush=True)
        win_tag = rank[0][0]
        dec["spec_type"] = A1[win_tag]
        dec["spec_type_tag"] = win_tag
        _save_state(self.state)
        spec_type = dec["spec_type"]
        has_mtp = "draft-mtp" in spec_type
        has_ngram = "ngram-mod" in spec_type
        print(f"A1 winner: {win_tag} spec-type={spec_type}", flush=True)

        # ---- Stage A2: spec-draft-p-min ----
        best_pmin = 0.2
        if has_mtp:
            print("\n########## STAGE A2: spec-draft-p-min ##########", flush=True)
            A2 = {
                "A2_pmin0.1": 0.1,
                "A2_pmin0.3": 0.3,
            }
            for tag, pm in A2.items():
                self.run_config(tag, self._cfg(**{"spec-type": spec_type,
                                                   "spec-draft-p-min": pm}), MID)
            pmin_tags = {dec["spec_type_tag"]: 0.2, "A2_pmin0.1": 0.1, "A2_pmin0.3": 0.3}
            rankp = self._best(list(pmin_tags))
            print("A2 ranking:", [(t, round(s[0], 2)) for t, s in rankp], flush=True)
            best_pmin = pmin_tags[rankp[0][0]]
            print(f"A2 winner: p-min={best_pmin}", flush=True)
        dec["pmin"] = best_pmin
        _save_state(self.state)

        # ---- Stage A3: ngram params ----
        best_ngram = {"spec-ngram-mod-n-match": 8, "spec-ngram-mod-n-min": 8,
                      "spec-ngram-mod-n-max": 16}
        if has_ngram:
            print("\n########## STAGE A3: ngram-mod params ##########", flush=True)
            common = {"spec-type": spec_type, "spec-draft-p-min": best_pmin}
            A3 = {
                "A3_nmax8":  {"spec-ngram-mod-n-max": 8},
                "A3_nmax32": {"spec-ngram-mod-n-max": 32},
                "A3_match4": {"spec-ngram-mod-n-match": 4},
            }
            for tag, extra in A3.items():
                self.run_config(tag, self._cfg(**common, **extra), MID)
            # reference for nmax=16/match=8 at best p-min
            if best_pmin == 0.2:
                ref = dec["spec_type_tag"]
            elif best_pmin == 0.1:
                ref = "A2_pmin0.1"
            else:
                ref = "A2_pmin0.3"
            ngram_for = {
                ref: {"spec-ngram-mod-n-match": 8, "spec-ngram-mod-n-min": 8, "spec-ngram-mod-n-max": 16},
                "A3_nmax8":  {"spec-ngram-mod-n-match": 8, "spec-ngram-mod-n-min": 8, "spec-ngram-mod-n-max": 8},
                "A3_nmax32": {"spec-ngram-mod-n-match": 8, "spec-ngram-mod-n-min": 8, "spec-ngram-mod-n-max": 32},
                "A3_match4": {"spec-ngram-mod-n-match": 4, "spec-ngram-mod-n-min": 8, "spec-ngram-mod-n-max": 16},
            }
            rankn = self._best(list(ngram_for))
            print("A3 ranking:", [(t, round(s[0], 2)) for t, s in rankn], flush=True)
            best_ngram = ngram_for[rankn[0][0]]
            print(f"A3 winner: {best_ngram}", flush=True)
        dec["ngram"] = best_ngram
        _save_state(self.state)

        spec_winner = self._cfg(**{"spec-type": spec_type, "spec-draft-p-min": best_pmin},
                                **best_ngram)

        # ---- Stage A4: draft backend sampling (GPU draft sampling) ----
        # Only meaningful when MTP drafting is active. Tests offloading the draft
        # head's sampling to the GPU vs CPU. Folded into spec_winner if it wins.
        draft_backend = None
        if has_mtp:
            print("\n########## STAGE A4: spec-draft-backend-sampling ##########", flush=True)
            self.run_config("A4_dbs_on",
                            dict(spec_winner, **{"spec-draft-backend-sampling": "true"}), MID)
            self.run_config("A4_dbs_off",
                            dict(spec_winner, **{"spec-draft-backend-sampling": "false"}), MID)
            ra = self._best(["A4_dbs_on", "A4_dbs_off"])
            print("A4 ranking:", [(t, round(s[0], 2)) for t, s in ra], flush=True)
            draft_backend = "true" if (not ra or ra[0][0] == "A4_dbs_on") else "false"
            spec_winner = dict(spec_winner, **{"spec-draft-backend-sampling": draft_backend})
            print(f"A4 winner: spec-draft-backend-sampling={draft_backend}", flush=True)
        dec["draft_backend"] = draft_backend
        _save_state(self.state)

        # ---- Stage B: 160k validation (+ triattention-interval at long ctx) ----
        # triattention-interval governs CPU-side eviction-scoring frequency; its
        # decode-stall cost only shows up past the window at long context, so it
        # is evaluated here rather than at 25k.
        print("\n########## STAGE B: 160k validation ##########", flush=True)
        self.run_config("B_winner", spec_winner, LONG)
        self.run_config("B_baseline", self._cfg(), LONG)
        self.run_config("B_winner_tri256",
                        dict(spec_winner, **{"triattention-interval": 256}), LONG)
        rb = self._best(["B_winner", "B_baseline", "B_winner_tri256"], ctx="160k")
        print("B ranking (160k):", [(t, round(s[0], 2)) for t, s in rb], flush=True)
        best_b = rb[0][0] if rb else "B_winner"
        if best_b == "B_baseline":
            spec_final = self._cfg()
        elif best_b == "B_winner_tri256":
            spec_final = dict(spec_winner, **{"triattention-interval": 256})
        else:
            spec_final = spec_winner
        dec["spec_final"] = spec_final
        _save_state(self.state)

        # ---- Stage C: ctx / parallel ----
        print("\n########## STAGE C: ctx-size vs parallel slots ##########", flush=True)
        for name, cc in C.CTX_PARALLEL_CONFIGS.items():
            conc = cc.get("parallel", 1)
            meas = [("90k", "text", "text_90k", conc), ("90k", "code", "code_90k", conc)]
            params = dict(spec_final)
            params.update(cc)
            self.run_config(f"C_{name}", params, meas)
        rc = self._best([f"C_{n}" for n in C.CTX_PARALLEL_CONFIGS], ctx="90k")
        print("C ranking (90k aggregate tg):",
              [(t, round(s[0], 2)) for t, s in rc], flush=True)
        ctx_winner_tag = rc[0][0] if rc else "C_single196"
        ctx_winner = C.CTX_PARALLEL_CONFIGS[ctx_winner_tag.removeprefix("C_")]
        dec["ctx_winner"] = ctx_winner_tag
        _save_state(self.state)

        # ---- Apply final ----
        final = dict(spec_final)
        final.update(ctx_winner)
        harness.set_params(final)
        harness.restart()
        harness.wait_ready()
        summary = {
            "spec_final": spec_final,
            "ctx_winner": ctx_winner_tag,
            "ctx_params": ctx_winner,
            "applied": final,
            "decisions": dec,
            "ttft_s": {
                "160k_winner": _ttft(best_b, "160k"),
                "90k_winner": _ttft(ctx_winner_tag, "90k"),
            },
        }
        C.SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
        print("\n########## SWEEP COMPLETE ##########", flush=True)
        print(json.dumps(summary, indent=2), flush=True)
        print(f"\nApplied config is live. Summary: {C.SUMMARY_JSON}", flush=True)
        print("NOTE: models.ini values updated; review the explanatory comments "
              "in config/models.ini to match the new values.", flush=True)
