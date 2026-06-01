# End-to-End Pipeline Audit

- Timestamp: 2026-06-01T09:07:01
- Status: PASS
- Radar source order: range,doppler,elevation,power
- Radar target order: range,elevation,velocity,power
- Samples: 16
- Max raw channel abs error: 0.000000
- Max dataloader radar abs diff: 0.000000
- Max dataloader box abs diff: 0.000020
- Model radar input channels: 4
- Model radar initial output channels: 4
- Letterbox box roundtrip error: 0.000122
- COCO identity mAP50-95: 100.000

This audit verifies that stored radar maps, loader channel reordering, dataloader outputs, model input contracts, detection postprocessing, and metric helpers use the same geometry and channel semantics.
