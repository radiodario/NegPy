import Foundation

/// The desktop's sync folder layout (`negpy/services/sync/store.py`): `edits/<hash>.json`,
/// `proxies/<hash>.png`, `rolls/<id>.json`. Hashes with `#` or `:` are spelled with `+` and
/// `-` in file names, as the desktop spells them.
struct EditRecord {
    static let version = 1
    var hash: String
    var fileName: String
    var modifiedAt: Double
    var settings: [String: Any]
    var mark: String?
    var device: String

    init(hash: String, fileName: String, modifiedAt: Double, settings: [String: Any], mark: String? = nil, device: String = SyncFolder.deviceName) {
        self.hash = hash
        self.fileName = fileName
        self.modifiedAt = modifiedAt
        self.settings = settings
        self.mark = mark
        self.device = device
    }

    init?(json: [String: Any]) {
        guard let hash = json["hash"] as? String else { return nil }
        self.hash = hash
        fileName = json["file_name"] as? String ?? ""
        modifiedAt = (json["modified_at"] as? NSNumber)?.doubleValue ?? 0
        settings = json["settings"] as? [String: Any] ?? [:]
        mark = json["mark"] as? String
        device = json["device"] as? String ?? ""
    }

    var json: [String: Any] {
        [
            "version": EditRecord.version,
            "hash": hash,
            "file_name": fileName,
            "modified_at": modifiedAt,
            "device": device,
            "mark": mark as Any,
            "settings": settings,
        ]
    }
}

struct RollRecord: Identifiable {
    var id: String
    var name: String
    var modifiedAt: Double
    var frames: [FrameRef]

    init?(json: [String: Any]) {
        guard let id = json["id"] as? String else { return nil }
        self.id = id
        name = json["name"] as? String ?? id
        modifiedAt = (json["modified_at"] as? NSNumber)?.doubleValue ?? 0
        frames = (json["frames"] as? [[String: Any]] ?? []).compactMap { FrameRef(json: $0) }
    }
}

struct FrameRef: Identifiable, Hashable {
    var hash: String
    var fileName: String
    var id: String { hash }

    init?(json: [String: Any]) {
        guard let hash = json["hash"] as? String else { return nil }
        self.hash = hash
        fileName = json["file_name"] as? String ?? hash
    }

    init(hash: String, fileName: String) {
        self.hash = hash
        self.fileName = fileName
    }
}

enum SyncError: Error {
    case notDownloaded
    case unreadable(String)
}

final class SyncFolder {
    static let deviceName = ProcessInfo.processInfo.hostName
    let root: URL

    init(root: URL) {
        self.root = root
    }

    var editsDir: URL { root.appendingPathComponent("edits", isDirectory: true) }
    var proxiesDir: URL { root.appendingPathComponent("proxies", isDirectory: true) }
    var rollsDir: URL { root.appendingPathComponent("rolls", isDirectory: true) }

    static func stem(_ hash: String) -> String {
        hash.replacingOccurrences(of: "#", with: "+").replacingOccurrences(of: ":", with: "-").replacingOccurrences(of: "/", with: "-")
    }

    func recordURL(_ hash: String) -> URL { editsDir.appendingPathComponent("\(SyncFolder.stem(hash)).json") }
    func proxyURL(_ hash: String) -> URL { proxiesDir.appendingPathComponent("\(SyncFolder.stem(hash)).png") }

    /// Reads a file through the file coordinator, asking iCloud to download it first when it
    /// is not local yet. Returns nil while the download is still pending.
    private func readCoordinated(_ url: URL) throws -> Data? {
        let values = try? url.resourceValues(forKeys: [.ubiquitousItemDownloadingStatusKey])
        if let status = values?.ubiquitousItemDownloadingStatus, status != .current {
            try? FileManager.default.startDownloadingUbiquitousItem(at: url)
            if status == .notDownloaded { return nil }
        }
        var data: Data?
        var coordinationError: NSError?
        NSFileCoordinator().coordinate(readingItemAt: url, options: [], error: &coordinationError) { readURL in
            data = try? Data(contentsOf: readURL)
        }
        if let coordinationError { throw SyncError.unreadable(coordinationError.localizedDescription) }
        return data
    }

    private func writeCoordinated(_ data: Data, to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        var coordinationError: NSError?
        var writeError: Error?
        NSFileCoordinator().coordinate(writingItemAt: url, options: .forReplacing, error: &coordinationError) { writeURL in
            do { try data.write(to: writeURL, options: .atomic) } catch { writeError = error }
        }
        if let coordinationError { throw SyncError.unreadable(coordinationError.localizedDescription) }
        if let writeError { throw writeError }
    }

    private func listJSON(_ dir: URL) -> [URL] {
        let urls = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        return urls.filter { $0.pathExtension == "json" || $0.lastPathComponent.hasSuffix(".json.icloud") }
    }

    func rolls() -> [RollRecord] {
        listJSON(rollsDir).compactMap { url -> RollRecord? in
            guard let data = try? readCoordinated(url),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
            return RollRecord(json: json)
        }
        .sorted { $0.modifiedAt > $1.modifiedAt }
    }

    /// Every frame with a record or a proxy, for a folder the Mac has not grouped into rolls.
    func looseFrames() -> [FrameRef] {
        var seen = Set<String>()
        var out: [FrameRef] = []
        for record in allEdits() where !seen.contains(record.hash) {
            seen.insert(record.hash)
            out.append(FrameRef(hash: record.hash, fileName: record.fileName))
        }
        return out
    }

    /// The newest record for the hash across the conflict copies a sync service names with a suffix.
    func edit(for hash: String) -> EditRecord? {
        let stem = SyncFolder.stem(hash)
        var newest: EditRecord?
        for url in listJSON(editsDir) where url.lastPathComponent.hasPrefix(stem) {
            guard let data = try? readCoordinated(url),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let record = EditRecord(json: json), record.hash == hash else { continue }
            if newest == nil || record.modifiedAt > newest!.modifiedAt { newest = record }
        }
        return newest
    }

    func allEdits() -> [EditRecord] {
        var byHash: [String: EditRecord] = [:]
        for url in listJSON(editsDir) {
            guard let data = try? readCoordinated(url),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let record = EditRecord(json: json) else { continue }
            if let have = byHash[record.hash], have.modifiedAt >= record.modifiedAt { continue }
            byHash[record.hash] = record
        }
        return Array(byHash.values)
    }

    func save(_ record: EditRecord) throws {
        let data = try JSONSerialization.data(withJSONObject: record.json, options: [.sortedKeys])
        try writeCoordinated(data, to: recordURL(record.hash))
    }

    /// The proxy's bytes, or nil while iCloud is still downloading it.
    func proxy(for hash: String) throws -> Data? {
        let url = proxyURL(hash)
        guard FileManager.default.fileExists(atPath: url.path) || FileManager.default.fileExists(atPath: url.path + ".icloud") else {
            return nil
        }
        return try readCoordinated(url)
    }
}
