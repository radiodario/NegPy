import CoreGraphics
import Foundation
import UIKit

/// The Rust pipeline through its Swift bindings (Generated/swift, from negpy_mobile's ffi).
enum Renderer {
    static func render(frame: Frame, edit: EditState, maxEdge: UInt32) throws -> UIImage {
        let json = try edit.json()
        let out = try frame.render(editJson: json, maxEdge: maxEdge)
        return image(width: Int(out.width), height: Int(out.height), rgba: out.rgba)
    }

    /// An sRGB-tagged image of the display-encoded RGBA bytes. The pipeline encodes with the
    /// Adobe RGB (1998) curve on Adobe RGB primaries, so the image is tagged as Adobe RGB.
    static func image(width: Int, height: Int, rgba: Data) -> UIImage {
        let space = CGColorSpace(name: CGColorSpace.adobeRGB1998) ?? CGColorSpaceCreateDeviceRGB()
        let provider = CGDataProvider(data: rgba as CFData)!
        let cg = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: width * 4,
            space: space,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        )!
        return UIImage(cgImage: cg)
    }
}
