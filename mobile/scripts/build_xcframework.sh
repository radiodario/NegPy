#!/usr/bin/env bash
# Builds negpy_mobile for iPhone and the simulator, generates the Swift bindings and packs
# NegPyCoreFFI.xcframework for the Xcode project. Run on a Mac with Xcode and rustup:
#   rustup target add aarch64-apple-ios aarch64-apple-ios-sim
#   mobile/scripts/build_xcframework.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CRATE="$ROOT/mobile/negpy_mobile"
OUT="$ROOT/mobile/ios/Generated"
LIB=libnegpy_mobile.a
PROFILE=${PROFILE:-release}

cd "$CRATE"
for target in aarch64-apple-ios aarch64-apple-ios-sim; do
  cargo build --$PROFILE --features ffi --target "$target"
done

rm -rf "$OUT"
mkdir -p "$OUT/swift" "$OUT/headers"
cargo run --$PROFILE --features ffi --bin uniffi-bindgen -- generate \
  --library "target/aarch64-apple-ios/$PROFILE/$LIB" --language swift --out-dir "$OUT/swift"
mv "$OUT/swift"/*.h "$OUT/headers/"
mv "$OUT/swift"/*.modulemap "$OUT/headers/module.modulemap"

rm -rf "$OUT/NegPyCoreFFI.xcframework"
xcodebuild -create-xcframework \
  -library "target/aarch64-apple-ios/$PROFILE/$LIB" -headers "$OUT/headers" \
  -library "target/aarch64-apple-ios-sim/$PROFILE/$LIB" -headers "$OUT/headers" \
  -output "$OUT/NegPyCoreFFI.xcframework"
echo "wrote $OUT"
