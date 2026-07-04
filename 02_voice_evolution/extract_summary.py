# -*- coding: utf-8 -*-
import os, json
r = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "report_evolution_campp.json"), encoding="utf-8"))
ents, kinds, dt, tta = r["entities"], r["kinds"], r["dt_months"], r["theta_to_anchor"]
ix = {e: i for i, e in enumerate(ents)}
fp = r["floor"]["per_epoch"]

print("=== FLOOR (same-period, cross-session) ===")
print("  NEW: %.3f +/- %.3f rad" % (r["floor"]["mean"], r["floor"]["std"]))
print("  OLD: 0.597 +/- 0.148 rad")
fl = [v["mean"] for v in fp.values()]
print("  per-epoch within-floor: min %.3f  max %.3f  mean %.3f" % (min(fl), max(fl), sum(fl)/len(fl)))
print("\n=== CEILING / landmarks ===")
print("  ceiling:", {k: round(v[0], 3) for k, v in r["ceiling"].items()})
for n in ("Echo", "Fries"):
    if n in ix:
        print("  %s theta_to_now: %.3f" % (n, tta[ix[n]][0]))

print("\n=== per epoch ===")
print("epoch    dt_mo  theta_to_now    within_floor  drop  n_after")
for i, e in enumerate(ents):
    if kinds[i] in ("epoch", "anchor"):
        wf = fp.get(e, {}).get("mean", float("nan"))
        pool = r["pool"][e]
        print("%-8s %6.1f  %.3f+-%.3f    %.3f        %d     %d"
              % (e, dt[i], tta[i][0], tta[i][1], wf, pool.get("dropped", 0), pool["n_after"]))

# old-vs-new epoch means (recent vs old)
rec = [tta[i][0] for i in range(len(ents)) if kinds[i] == "epoch" and dt[i] is not None and dt[i] <= 24]
old = [tta[i][0] for i in range(len(ents)) if kinds[i] == "epoch" and dt[i] is not None and dt[i] >= 30]
print("\nrecent(<=24mo) mean theta_to_now: %.3f   older(>=30mo): %.3f   gap: %.3f"
      % (sum(rec)/len(rec), sum(old)/len(old), sum(old)/len(old) - sum(rec)/len(rec)))
