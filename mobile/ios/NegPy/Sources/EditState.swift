import Foundation

/// One frame's edit as the desktop stores it: a flat dictionary keyed by field name. The
/// typed accessors cover the controls this app shows; every other key rides along untouched.
struct EditState {
    var hash: String
    var fileName: String
    var settings: [String: Any]
    var mark: String?

    static func defaults() -> [String: Any] {
        let text = defaultEditJson()
        return (try? JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any]) ?? [:]
    }

    func json() throws -> String {
        let data = try JSONSerialization.data(withJSONObject: settings, options: [.sortedKeys])
        return String(decoding: data, as: UTF8.self)
    }

    func double(_ key: String, _ fallback: Double = 0) -> Double {
        (settings[key] as? NSNumber)?.doubleValue ?? fallback
    }

    func bool(_ key: String, _ fallback: Bool = false) -> Bool {
        (settings[key] as? Bool) ?? ((settings[key] as? NSNumber)?.boolValue ?? fallback)
    }

    func string(_ key: String, _ fallback: String = "") -> String {
        settings[key] as? String ?? fallback
    }

    func int(_ key: String, _ fallback: Int = 0) -> Int {
        (settings[key] as? NSNumber)?.intValue ?? fallback
    }

    mutating func set(_ key: String, _ value: Any) {
        settings[key] = value
    }
}

/// A slider the editor shows: its key in the flat dictionary, its label and its range.
struct SliderSpec: Identifiable {
    let key: String
    let label: String
    let range: ClosedRange<Double>
    let step: Double
    let neutral: Double
    var id: String { key }

    static let exposure: [SliderSpec] = [
        SliderSpec(key: "density", label: "Print Density", range: 0.0...2.0, step: 0.01, neutral: 1.0),
        SliderSpec(key: "grade", label: "Grade", range: 50...180, step: 1, neutral: 115),
        SliderSpec(key: "wb_cyan", label: "Cyan", range: -1...1, step: 0.01, neutral: 0),
        SliderSpec(key: "wb_magenta", label: "Magenta", range: -1...1, step: 0.01, neutral: 0),
        SliderSpec(key: "wb_yellow", label: "Yellow", range: -1...1, step: 0.01, neutral: 0),
        SliderSpec(key: "shadow_density", label: "Shadows", range: -0.5...0.5, step: 0.01, neutral: 0),
        SliderSpec(key: "highlight_density", label: "Highlights", range: -0.5...0.5, step: 0.01, neutral: 0),
        SliderSpec(key: "toe", label: "Toe", range: 0...1, step: 0.01, neutral: 0),
        SliderSpec(key: "shoulder", label: "Shoulder", range: 0...1, step: 0.01, neutral: 0),
        SliderSpec(key: "cast_removal_strength", label: "Cast Removal", range: 0...1, step: 0.01, neutral: 0.5),
        SliderSpec(key: "contrast_mask_strength", label: "Contrast Mask", range: 0...1, step: 0.01, neutral: 0),
    ]

    static let lab: [SliderSpec] = [
        SliderSpec(key: "saturation", label: "Saturation", range: 0...2, step: 0.01, neutral: 1),
        SliderSpec(key: "sharpen", label: "Sharpen", range: 0...1, step: 0.01, neutral: 0.25),
        SliderSpec(key: "clahe_strength", label: "Local Contrast", range: 0...1, step: 0.01, neutral: 0),
        SliderSpec(key: "glow_amount", label: "Glow", range: 0...1, step: 0.01, neutral: 0),
    ]

    static let toning: [SliderSpec] = [
        SliderSpec(key: "sepia_strength", label: "Sepia", range: 0...1, step: 0.01, neutral: 0),
        SliderSpec(key: "selenium_strength", label: "Selenium", range: 0...1, step: 0.01, neutral: 0),
        SliderSpec(key: "gold_strength", label: "Gold", range: 0...1, step: 0.01, neutral: 0),
        SliderSpec(key: "blue_strength", label: "Blue", range: 0...1, step: 0.01, neutral: 0),
    ]

    static let finish: [SliderSpec] = [
        SliderSpec(key: "vignette_stops", label: "Vignette", range: -2...2, step: 0.01, neutral: 0),
        SliderSpec(key: "fine_rotation", label: "Straighten", range: -45...45, step: 0.1, neutral: 0),
    ]
}

enum FilmMode: String, CaseIterable, Identifiable {
    case color = "Color Negative"
    case bw = "B&W Negative"
    var id: String { rawValue }
    var label: String { self == .color ? "Color" : "B&W" }
}
