import SwiftUI

/// One frame: the print on top, the controls below. Every change renders a preview and
/// saves the edit after a short pause, so the Mac picks it up on its next sync pass.
struct EditorView: View {
    @EnvironmentObject var library: Library
    let frame: FrameRef

    @State private var edit: EditState?
    @State private var image: UIImage?
    @State private var rendering = false
    @State private var error: String?
    @State private var saveTask: Task<Void, Never>?
    @State private var renderGeneration = 0

    var body: some View {
        VStack(spacing: 0) {
            ZStack {
                Color.black
                if let image {
                    Image(uiImage: image).resizable().scaledToFit()
                } else if let error {
                    Text(error).foregroundStyle(.red).padding()
                } else {
                    ProgressView()
                }
            }
            .frame(maxWidth: .infinity)
            .frame(height: 360)
            if edit != nil {
                controls
            }
        }
        .navigationTitle(frame.fileName)
        .navigationBarTitleDisplayMode(.inline)
        .task {
            edit = library.edit(for: frame.hash)
            render()
        }
    }

    private var controls: some View {
        Form {
            Section("Film") {
                Picker("Film", selection: binding(for: "process_mode", fallback: FilmMode.color.rawValue)) {
                    ForEach(FilmMode.allCases) { mode in Text(mode.label).tag(mode.rawValue) }
                }
                .pickerStyle(.segmented)
                Toggle("Auto Density", isOn: boolBinding("auto_exposure"))
                Toggle("Auto Grade", isOn: boolBinding("auto_normalize_contrast"))
            }
            sliderSection("Exposure", SliderSpec.exposure)
            sliderSection("Lab", SliderSpec.lab)
            sliderSection("Toning", SliderSpec.toning)
            sliderSection("Finish", SliderSpec.finish)
            Section("Geometry") {
                HStack {
                    Button { rotate(by: 1) } label: { Label("Rotate Left", systemImage: "rotate.left") }
                    Spacer()
                    Button { rotate(by: -1) } label: { Label("Rotate Right", systemImage: "rotate.right") }
                }
                .buttonStyle(.bordered)
                Toggle("Flip Horizontal", isOn: boolBinding("flip_horizontal"))
                Toggle("Flip Vertical", isOn: boolBinding("flip_vertical"))
            }
            Section("Triage") {
                Picker("Mark", selection: markBinding) {
                    Text("None").tag("")
                    Text("Keeper").tag("keeper")
                    Text("Excluded").tag("excluded")
                }
                .pickerStyle(.segmented)
            }
        }
    }

    private func sliderSection(_ title: String, _ specs: [SliderSpec]) -> some View {
        Section(title) {
            ForEach(specs) { spec in
                VStack(alignment: .leading, spacing: 2) {
                    HStack {
                        Text(spec.label)
                        Spacer()
                        Text(format(value(spec), spec)).monospacedDigit().foregroundStyle(.secondary)
                    }
                    Slider(value: doubleBinding(spec.key, fallback: spec.neutral), in: spec.range, step: spec.step)
                        .onTapGesture(count: 2) { update(spec.key, spec.neutral) }
                }
            }
        }
    }

    private func value(_ spec: SliderSpec) -> Double {
        edit?.double(spec.key, spec.neutral) ?? spec.neutral
    }

    private func format(_ v: Double, _ spec: SliderSpec) -> String {
        spec.step >= 1 ? String(format: "%.0f", v) : String(format: "%.2f", v)
    }

    private func doubleBinding(_ key: String, fallback: Double) -> Binding<Double> {
        Binding(get: { edit?.double(key, fallback) ?? fallback }, set: { update(key, $0) })
    }

    private func boolBinding(_ key: String) -> Binding<Bool> {
        Binding(get: { edit?.bool(key) ?? false }, set: { update(key, $0) })
    }

    private func binding(for key: String, fallback: String) -> Binding<String> {
        Binding(get: { edit?.string(key, fallback) ?? fallback }, set: { update(key, $0) })
    }

    private var markBinding: Binding<String> {
        Binding(get: { edit?.mark ?? "" }, set: { value in
            edit?.mark = value.isEmpty ? nil : value
            scheduleSave()
        })
    }

    private func rotate(by quarter: Int) {
        let current = edit?.int("rotation") ?? 0
        update("rotation", ((current + quarter) % 4 + 4) % 4)
    }

    private func update(_ key: String, _ value: Any) {
        edit?.set(key, value)
        render()
        scheduleSave()
    }

    private func scheduleSave() {
        saveTask?.cancel()
        saveTask = Task {
            try? await Task.sleep(for: .milliseconds(800))
            guard !Task.isCancelled, let edit else { return }
            library.save(edit)
        }
    }

    private func render() {
        guard let edit else { return }
        renderGeneration += 1
        let generation = renderGeneration
        let maxEdge = UInt32(UIScreen.main.nativeBounds.width)
        Task.detached(priority: .userInitiated) {
            do {
                guard let frame = try await MainActor.run(body: { try library.frame(for: edit.hash) }) else {
                    await MainActor.run { error = "Waiting for iCloud to download this frame." }
                    return
                }
                let rendered = try Renderer.render(frame: frame, edit: edit, maxEdge: maxEdge)
                await MainActor.run {
                    if generation == renderGeneration {
                        image = rendered
                        error = nil
                    }
                }
            } catch {
                await MainActor.run { self.error = "\(error)" }
            }
        }
    }
}
