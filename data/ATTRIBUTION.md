# Attribution

## Source

**Open Power System Data** — Time series data package, hourly resolution.

- Package `opsd_time_series`, snapshot **`2020-10-06`**
- Landing page: https://data.open-power-system-data.org/time_series/2020-10-06
- File used: `time_series_60min_singleindex.csv` — 130,339,665 bytes, sha256
  `6a7f2bc571314cbf9c321cc03437691cd4be95c3a6f075e60ff99e8035c704c8`
- OPSD's own derivation notebook:
  https://github.com/Open-Power-System-Data/datapackage_timeseries/blob/2020-10-06/main.ipynb
- Package author: Jonathan Muehlenpfordt, Neon Neue Energieökonomik

**Upstream source of the series used here: the ENTSO-E Transparency Platform.**
OPSD republishes it; it did not originate the measurements.

The snapshot is pinned rather than tracking `latest` because OPSD stopped
publishing after 2020-10-06. `data/download_opsd.py` verifies the checksum above
and fails loudly if upstream ever changes, since the committed slice could then
no longer be reproduced from it.

## On the licence

The `datapackage.json` for this snapshot carries **no machine-readable licence
field**. That was checked rather than assumed — its top-level keys are `profile`,
`name`, `id`, `title`, `description`, `longDescription`, `homepage`,
`documentation`, `version`, `created`, `lastChanges`, `keywords`,
`geographicalScope`, `temporal`, `contributors`, `sources` and `resources`, and
nothing in the descriptive text states terms.

So no licence is asserted here. If you intend to redistribute this data or use it
commercially, read the terms at the OPSD landing page above and at the ENTSO-E
Transparency Platform first. The derived slice in `data/processed/` is committed
on the basis that it is a small, transformed subset used for research and
demonstration with the source credited — not on the basis of a licence grant this
repository has verified.

The **code** here is MIT licensed (see `LICENSE`). That covers the code only, not
the data.

## What was changed

`data/processed/site_hourly.csv` is a derived subset, not a copy:

1. sliced to 2018-10-01 → 2020-09-30, and to four of the source's 300 columns;
2. national load and solar aggregates rescaled to a hypothetical 1.0 MW / 0.8 MWp
   site by dividing by the in-window maximum;
3. gaps of up to three hours interpolated, and every filled row flagged.

Full detail in `DATA_DICTIONARY.md`. The transformation is reproducible:

```bash
python data/download_opsd.py && python data/prepare_dataset.py
```
