import SwiftUI

struct RollListView: View {
    @EnvironmentObject var library: Library

    var body: some View {
        List {
            if library.rolls.isEmpty && library.loose.isEmpty {
                Text("No frames yet. Open a roll in NegPy on the Mac with Sync Edits and Write Phone Previews on.")
                    .foregroundStyle(.secondary)
            }
            ForEach(library.rolls) { roll in
                NavigationLink(value: roll.id) {
                    HStack {
                        Image(systemName: "film.stack")
                        Text(roll.name)
                        Spacer()
                        Text("\(roll.frames.count)").foregroundStyle(.secondary)
                    }
                }
            }
            if !library.loose.isEmpty {
                NavigationLink(value: "loose") {
                    HStack {
                        Image(systemName: "photo.on.rectangle")
                        Text("Other Frames")
                        Spacer()
                        Text("\(library.loose.count)").foregroundStyle(.secondary)
                    }
                }
            }
        }
        .navigationTitle("Rolls")
        .navigationDestination(for: String.self) { id in
            if id == "loose" {
                FrameGridView(title: "Other Frames", frames: library.loose)
            } else if let roll = library.rolls.first(where: { $0.id == id }) {
                FrameGridView(title: roll.name, frames: roll.frames)
            }
        }
        .refreshable { library.refresh() }
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Menu {
                    Button("Refresh") { library.refresh() }
                    Button("Change Sync Folder…", role: .destructive) { library.forget() }
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
            }
        }
        .alert("NegPy", isPresented: Binding(get: { library.message != nil }, set: { if !$0 { library.message = nil } })) {
            Button("OK") { library.message = nil }
        } message: {
            Text(library.message ?? "")
        }
    }
}

struct FrameGridView: View {
    @EnvironmentObject var library: Library
    let title: String
    let frames: [FrameRef]

    private let columns = [GridItem(.adaptive(minimum: 150), spacing: 8)]

    var body: some View {
        ScrollView {
            LazyVGrid(columns: columns, spacing: 8) {
                ForEach(frames) { frame in
                    NavigationLink(value: frame) {
                        ThumbnailCell(frame: frame)
                    }
                }
            }
            .padding(8)
        }
        .navigationTitle(title)
        .navigationDestination(for: FrameRef.self) { frame in
            EditorView(frame: frame)
        }
    }
}

struct ThumbnailCell: View {
    @EnvironmentObject var library: Library
    let frame: FrameRef

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            ZStack {
                Rectangle().fill(Color(white: 0.15))
                if let image = library.thumbnail(for: frame.hash) {
                    Image(uiImage: image).resizable().scaledToFit()
                } else {
                    VStack(spacing: 6) {
                        ProgressView()
                        Text("Waiting for iCloud").font(.caption2).foregroundStyle(.secondary)
                    }
                }
            }
            .aspectRatio(3 / 2, contentMode: .fit)
            .clipShape(RoundedRectangle(cornerRadius: 6))
            Text(frame.fileName).font(.caption).lineLimit(1).foregroundStyle(.secondary)
        }
    }
}
