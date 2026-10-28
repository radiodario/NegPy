import Foundation
import SwiftUI

/// The app's session: the sync folder, its rolls, and the frames' decoded proxies.
@MainActor
final class Library: ObservableObject {
    @Published private(set) var folder: SyncFolder?
    @Published private(set) var rolls: [RollRecord] = []
    @Published private(set) var loose: [FrameRef] = []
    @Published var message: String?

    private let bookmarkKey = "sync_folder_bookmark"
    private var frames: [String: Frame] = [:]
    private var thumbnails: [String: UIImage] = [:]

    func restore() {
        guard folder == nil, let data = UserDefaults.standard.data(forKey: bookmarkKey) else { return }
        var stale = false
        if let url = try? URL(resolvingBookmarkData: data, options: [], relativeTo: nil, bookmarkDataIsStale: &stale) {
            open(url, remember: stale)
        }
    }

    func adopt(folder url: URL) {
        open(url, remember: true)
    }

    private func open(_ url: URL, remember: Bool) {
        guard url.startAccessingSecurityScopedResource() else {
            message = "NegPy cannot read that folder."
            return
        }
        if remember, let data = try? url.bookmarkData(options: [], includingResourceValuesForKeys: nil, relativeTo: nil) {
            UserDefaults.standard.set(data, forKey: bookmarkKey)
        }
        folder = SyncFolder(root: url)
        refresh()
    }

    func forget() {
        folder?.root.stopAccessingSecurityScopedResource()
        folder = nil
        rolls = []
        loose = []
        frames = [:]
        thumbnails = [:]
        UserDefaults.standard.removeObject(forKey: bookmarkKey)
    }

    func refresh() {
        guard let folder else { return }
        rolls = folder.rolls()
        let inRolls = Set(rolls.flatMap { $0.frames.map(\.hash) })
        loose = folder.looseFrames().filter { !inRolls.contains($0.hash) }
    }

    /// The frame's decoded proxy; nil while iCloud still downloads it.
    func frame(for hash: String) throws -> Frame? {
        if let have = frames[hash] { return have }
        guard let folder, let png = try folder.proxy(for: hash) else { return nil }
        let frame = try Frame.fromPng(png: png)
        frames[hash] = frame
        return frame
    }

    func edit(for hash: String) -> EditState {
        if let record = folder?.edit(for: hash) {
            return EditState(hash: hash, fileName: record.fileName, settings: record.settings, mark: record.mark)
        }
        return EditState(hash: hash, fileName: hash, settings: EditState.defaults(), mark: nil)
    }

    func save(_ edit: EditState) {
        guard let folder else { return }
        let record = EditRecord(hash: edit.hash, fileName: edit.fileName, modifiedAt: Date().timeIntervalSince1970, settings: edit.settings, mark: edit.mark)
        do {
            try folder.save(record)
            thumbnails[edit.hash] = nil
        } catch {
            message = "Could not save the edit: \(error.localizedDescription)"
        }
    }

    func thumbnail(for hash: String) -> UIImage? {
        if let have = thumbnails[hash] { return have }
        guard let frame = try? frame(for: hash) else { return nil }
        let edit = edit(for: hash)
        guard let image = try? Renderer.render(frame: frame, edit: edit, maxEdge: 320) else { return nil }
        thumbnails[hash] = image
        return image
    }
}
