# NegPy on iPhone and iPad

The phone app renders the same pipeline as the desktop, on the device, and shares the
desktop's edits through iCloud Drive.

## Parts

- `negpy_mobile/`: a Rust crate that ports the desktop CPU pipeline
  (`negpy/services/rendering/engine.py` and `negpy/features/*`) onto the `negpy_core`
  kernels. It reads the edit as the flat dict the desktop database stores. `tests/` holds
  parity tests against fixtures the desktop engine rendered; a stage passes when its
  pixels match bit for bit.
- `scripts/make_fixtures.py`: writes those fixtures. Run it after a pipeline change on the
  desktop, then run `cargo test` in `negpy_mobile/`.
- `scripts/build_xcframework.sh`: builds the crate for iPhone and the simulator and
  generates the Swift bindings (uniffi) into `ios/Generated/`.
- `ios/`: the SwiftUI app. `project.yml` is an xcodegen spec: `brew install xcodegen`,
  `xcodegen generate`, then open `NegPy.xcodeproj`.

## Sync

The desktop never syncs its database file: a synced drive moves files, not transactions.
Instead it writes one JSON record per frame into the sync folder (`negpy/services/sync/`),
keyed by the frame's content hash, plus a proxy of the frame: the baked scene-linear
source at a bounded size, as a 16-bit PNG. The phone reads the records and the proxies,
renders, and writes records of its own. The newest `modified_at` wins on both sides.

Turn it on in the desktop's Preferences, Sync. On the phone, open the same folder once.

## Build on a Mac

```bash
rustup target add aarch64-apple-ios aarch64-apple-ios-sim
mobile/scripts/build_xcframework.sh
cd mobile/ios && xcodegen generate && open NegPy.xcodeproj
```

## Not yet on the phone

- Slides (the transparency path) and the Flat master's export.
- Print stocks with a baked LUT (`negpy/features/exposure/print_luts/`): the engine
  reports the stock as unsupported and the app falls back to Digital.
- Source bakes (flat field, sensor unmix, dust repair) run on the desktop before the proxy.
- Auto Crop detection, retouching, stitching, HDR merges, export.
