import CoreAudio
import Foundation

struct RecordOptions {
    var outputRoot: URL
    var micQuery: String?
    var maxMinutes: Double?
    var sessionDirectory: URL?
}

/// Foreground recording loop: preflight, capture both sources, show live status,
/// warn on health problems, and finalize on Ctrl-C.
final class Recorder {
    private static let minimumFreeBytes: Int64 = 2_000_000_000
    private static let flushSeconds = 2.0

    private let options: RecordOptions
    private var stopRequested = false
    private let terminal = Terminal()

    init(options: RecordOptions) { self.options = options }

    func run() throws -> Int32 {
        // Preflight: nothing is reported as recording until every check passes.
        try Permissions.requireMicrophone()
        let systemAudioVerified = try Permissions.requireSystemAudio()
        let mic = try Devices.resolveInput(options.micQuery)
        let root = options.sessionDirectory?.deletingLastPathComponent() ?? options.outputRoot
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try requireFreeSpace(at: root)

        let session = try Session(root: root, directory: options.sessionDirectory)
        defer { session.close() }
        let tap = try SystemAudioTap()
        defer { tap.destroy() }

        let micTrack = try Track(
            name: "mic", deviceName: mic.name, deviceID: mic.id, sampleRate: mic.sampleRate,
            file: session.directory.appendingPathComponent("mic.wav"), checksFlatLine: true)
        let systemRate = try Devices.nominalSampleRate(tap.aggregateID)
        // System audio is legitimately digital silence when nothing plays, so it gets
        // no flat-line check; the inactivity window covers a dead tap instead.
        let systemTrack = try Track(
            name: "system", deviceName: "System output (all apps)", deviceID: tap.aggregateID,
            sampleRate: systemRate, file: session.directory.appendingPathComponent("system.wav"),
            checksFlatLine: false, expectsFrames: Devices.defaultOutputIsPlaying)
        let tracks = [micTrack, systemTrack]

        session.manifest["sources"] = tracks.map {
            ["name": $0.name, "device": $0.deviceName, "sample_rate": $0.sampleRate,
             "file": $0.writer.url.lastPathComponent]
        }
        session.manifest["system_audio_permission"] = systemAudioVerified ? "verified" : "unverified"
        session.log("INFO", "Session directory: \(session.directory.path)")
        session.log("INFO", "Microphone: \(mic.name) [\(mic.uid)] \(Int(mic.sampleRate)) Hz, \(mic.channels) ch")
        session.log("INFO", "System audio: global tap \(Int(systemRate)) Hz")
        if !systemAudioVerified {
            session.log("WARNING", "Could not verify the system audio permission")
        }

        for track in tracks { try track.capture.start() }
        let started = Date()
        session.manifest["state"] = "recording"
        session.manifest["recording_started_at"] = session.now()
        try session.saveManifest()
        session.log("INFO", "Recording started")

        terminal.printHeader(session: session, mic: mic, systemRate: systemRate, verified: systemAudioVerified)
        installStopHandlers()

        var lastTick = Date()
        var lastFlush = Date()
        var stopReason = "user"
        while !stopRequested {
            RunLoop.current.run(until: Date().addingTimeInterval(0.25))
            let now = Date()
            let elapsed = now.timeIntervalSince(lastTick)
            lastTick = now

            for track in tracks {
                for event in track.pump(elapsed: elapsed, since: started) { report(event, session: session) }
            }
            if now.timeIntervalSince(lastFlush) >= Self.flushSeconds {
                lastFlush = now
                for track in tracks { for event in track.flush() { report(event, session: session) } }
                if let free = freeBytes(at: session.directory), free < Self.minimumFreeBytes / 4 {
                    terminal.warn("Disk almost full (\(free / 1_000_000) MB left). Stop soon to keep the recording intact.")
                }
            }
            let duration = now.timeIntervalSince(started)
            terminal.status(duration: duration, tracks: tracks)

            if tracks.allSatisfy(\.failed) { stopReason = "all sources failed"; break }
            if tracks.contains(where: { $0.writer.isFull }) { stopReason = "file size limit"; break }
            if let limit = options.maxMinutes, duration >= limit * 60 { stopReason = "time limit"; break }
        }

        terminal.endStatus()
        var finishErrors: [String] = []
        for track in tracks {
            do { try track.finish(since: started) } catch {
                finishErrors.append("\(track.name): \(error)")
                session.log("ERROR", "Finalizing \(track.name) failed: \(error)")
            }
        }

        let warnings = tracks.reduce(0) { $0 + $1.health.warningCount }
        let outcome: String
        if !finishErrors.isEmpty || tracks.contains(where: \.failed) {
            outcome = "failed"
        } else if warnings > 0 {
            outcome = "degraded"
        } else {
            outcome = "complete"
        }
        session.manifest["state"] = "recorded"
        session.manifest["recording_outcome"] = outcome
        session.manifest["stop_reason"] = stopReason
        session.manifest["recording_stopped_at"] = session.now()
        session.manifest["duration_seconds"] = tracks.map(\.writer.durationSeconds).max() ?? 0
        session.manifest["health_warnings"] = warnings
        try session.saveManifest()
        session.log("INFO", "Recording stopped (\(stopReason)); outcome \(outcome); \(warnings) health warnings")

        terminal.printSummary(session: session, tracks: tracks, outcome: outcome, stopReason: stopReason, warnings: warnings)
        restoreSignals()
        return outcome == "failed" ? 1 : 0
    }

    private func report(_ event: HealthEvent, session: Session) {
        session.log(event.active ? "WARNING" : "INFO", "[\(event.source)] \(event.condition.rawValue): \(event.message)")
        session.manifest["last_health_event"] = "\(session.now()) \(event.source) \(event.condition.rawValue) \(event.active ? "raised" : "cleared")"
        if event.active { terminal.warn(event.message) } else { terminal.info(event.message) }
    }

    private func installStopHandlers() {
        for sig in [SIGINT, SIGTERM, SIGHUP] {
            signal(sig, SIG_IGN)
            let source = DispatchSource.makeSignalSource(signal: sig, queue: .main)
            source.setEventHandler { self.stopRequested = true }
            source.resume()
            signalSources.append(source)
        }
    }

    private var signalSources: [DispatchSourceSignal] = []

    private func restoreSignals() {
        for source in signalSources { source.cancel() }
        signalSources.removeAll()
        for sig in [SIGINT, SIGTERM, SIGHUP] { signal(sig, SIG_DFL) }
    }

    private func freeBytes(at url: URL) -> Int64? {
        let values = try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage
    }

    private func requireFreeSpace(at url: URL) throws {
        guard let free = freeBytes(at: url) else {
            throw SetupError("Cannot read free space for \(url.path).", fix: "Choose another folder with --out.")
        }
        if free < Self.minimumFreeBytes {
            throw SetupError(
                "Only \(free / 1_000_000) MB free at \(url.path); at least \(Self.minimumFreeBytes / 1_000_000) MB is required.",
                fix: "Free disk space or choose another folder with --out.")
        }
    }
}
