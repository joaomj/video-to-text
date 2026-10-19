import Foundation

/// A meeting session folder: audio tracks, a persistent log, and a manifest that
/// records state so later stages can resume.
final class Session {
    let id: String
    let directory: URL
    let logURL: URL
    let manifestURL: URL
    private let logHandle: FileHandle
    private let timestamp = ISO8601DateFormatter()
    var manifest: [String: Any]

    /// Uses `directory` when given (it must not exist yet), or a new timestamped folder under `root`.
    init(root: URL, directory requested: URL?) throws {
        if let requested {
            directory = requested
        } else {
            let formatter = DateFormatter()
            formatter.locale = Locale(identifier: "en_US_POSIX")
            formatter.dateFormat = "yyyy-MM-dd_HHmmss"
            directory = root.appendingPathComponent(formatter.string(from: Date()), isDirectory: true)
        }
        id = directory.lastPathComponent
        if FileManager.default.fileExists(atPath: directory.path) {
            throw SetupError(
                "Session folder \(directory.path) already exists.",
                fix: "Choose a new session folder; meetrec never overwrites a recording.")
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        logURL = directory.appendingPathComponent("recording.log")
        manifestURL = directory.appendingPathComponent("session.json")
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        logHandle = try FileHandle(forWritingTo: logURL)
        timestamp.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        manifest = ["session_id": id, "state": "starting", "created_at": timestamp.string(from: Date())]
        try saveManifest()
    }

    func log(_ level: String, _ message: String) {
        let line = "\(timestamp.string(from: Date())) [\(level)] [\(id)] \(message)\n"
        // A failed log write must not stop the recording; report it on stderr instead.
        do { try logHandle.write(contentsOf: Data(line.utf8)) } catch {
            FileHandle.standardError.write(Data("log write failed: \(error)\n".utf8))
        }
    }

    func now() -> String { timestamp.string(from: Date()) }

    /// Writes the manifest to a temporary file and renames it, so readers never see a partial file.
    func saveManifest() throws {
        let data = try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys])
        let temporary = directory.appendingPathComponent(".session.json.tmp")
        try data.write(to: temporary)
        if rename(temporary.path, manifestURL.path) != 0 {
            throw SetupError(
                "Cannot update \(manifestURL.path): \(String(cString: strerror(errno))).",
                fix: "Check free disk space and folder permissions.")
        }
    }

    func close() { try? logHandle.close() }
}
