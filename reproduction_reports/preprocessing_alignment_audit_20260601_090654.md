# Preprocessing Alignment Audit

- Timestamp: 2026-06-01T09:06:59
- Input shape: [320, 320]
- Radar source order: range,doppler,elevation,power
- Radar target order: range,elevation,velocity,power
- Samples: 16
- Mean CSV-to-NPZ match ratio: 1.0000
- Min CSV-to-NPZ match ratio: 1.0000
- Mean nearest distance px: 0.3548
- Max nearest distance px: 1.4142
- Mean raw channel abs error: 0.000000
- Max raw channel abs error: 0.000000
- Overlay directory: reproduction_reports/preprocess_visuals/20260601_090654
- Montage: 

A match ratio close to 1.0, nearest distances near 0, and raw channel errors near 0 indicate that the radar NPZ pixels loaded by training align with the official WaterScenes CSV `u/v` coordinates and keep the expected REVP channel semantics after the same image letterbox transform.
