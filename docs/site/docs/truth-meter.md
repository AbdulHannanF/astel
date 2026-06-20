# Truth Meter

The **Truth Meter** is Astel's per-asset quality report — and its trust brand. Where
other generators ship a confident-looking result with no accuracy signal, every
Astel asset carries an honest `astel.quality-report/v0` report that distinguishes
*measured* from *generated*, and reports `null` (with a reason) for anything it did
not actually measure.

## What it reports

- **Geometric error** — Chamfer distance of L3 vs. the L1/source reference, in mm,
  with method. Measured on the **capture path** (real photos/video → SfM → splats,
  validated against DTU ground truth). For purely **generated** assets there is no
  ground-truth scan, so this is honestly `null` with a reason — never fabricated.
- **Fidelity** — held-out-view PSNR/SSIM/LPIPS. On the generative path this is a
  *self-consistency / distillation* number (the L3 reproducing the L2's own renders),
  explicitly flagged as such, not accuracy versus a real object.
- **Scale confidence** — metres-per-unit with a confidence interval and method
  (SfM scale, VLM size estimate, or ungrounded identity). The interval is shown, and
  an ungrounded asset says so.
- **Hallucination / provenance** — the measured-vs-generated fraction, and (for
  capture inputs) which regions were *seen* by the camera versus completed by the
  generator. Completion over unseen regions is flagged in the confidence channel,
  never silently merged into measured reality.
- **Origin** — `measured`, `generated`, or `stub`, plus explicit caveats.

## Why `null` is a feature

The honesty contract (CLAUDE.md §1.3, §10.4) is binding: if a stage cannot meet an
accuracy target, the system reports the measured shortfall — or an explicit
"not measured" — rather than inventing a number. A Truth Meter full of honest
`null`s on a generated asset is the point: it tells you exactly which claims are
backed by measurement and which are not.

## In the app

The web app renders the live report with a mandatory origin pill (`MEASURED` /
`GENERATED` / `STUB`) and surfaces the caveats inline, so the asset's provenance is
never hidden behind a polished render.
