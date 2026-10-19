import Foundation

/// Live terminal output. Warnings print on their own lines above the status line,
/// in color and with a bell, so they are visible during the meeting.
final class Terminal {
    private let isTTY = isatty(STDOUT_FILENO) == 1
    private var statusVisible = false
    private var lastPlainStatus = Date.distantPast

    func printHeader(session: Session, mic: InputDevice, systemRate: Double, verified: Bool) {
        line("Recording session \(session.id)")
        line("  Folder:  \(session.directory.path)")
        line("  Mic:     \(mic.name) (\(Int(mic.sampleRate)) Hz)")
        line("  System:  all apps, current output device (\(Int(systemRate)) Hz)")
        if !verified {
            line("  Note:    system audio permission could not be verified; watch the system level.")
        }
        line("  Press Ctrl-C to stop and save.")
        line("")
    }

    func status(duration: Double, tracks: [Track]) {
        let parts = tracks.map { track -> String in
            let level = track.health.lastBlock.map { meter($0.rmsDb) } ?? "  waiting  "
            let flag = track.failed ? " FAILED" : (track.health.hasActiveProblems ? " !" : "")
            return "\(track.name) \(level)\(flag)"
        }
        let text = "\(clock(duration))  " + parts.joined(separator: "   ")
        if isTTY {
            write("\r\u{1B}[2K" + text)
            statusVisible = true
        } else if Date().timeIntervalSince(lastPlainStatus) >= 60 {
            lastPlainStatus = Date()
            line(text)
        }
    }

    func endStatus() {
        if statusVisible { write("\n") }
        statusVisible = false
    }

    func warn(_ message: String) {
        clearStatus()
        line(isTTY ? "\u{07}\u{1B}[1;33mWARNING\u{1B}[0m \(message)" : "WARNING \(message)")
    }

    func info(_ message: String) {
        clearStatus()
        line(isTTY ? "\u{1B}[32mOK\u{1B}[0m \(message)" : "OK \(message)")
    }

    func printSummary(session: Session, tracks: [Track], outcome: String, stopReason: String, warnings: Int) {
        line("")
        line("Recording \(outcome) (stopped: \(stopReason), \(warnings) health warning\(warnings == 1 ? "" : "s"))")
        for track in tracks {
            line("  \(track.name): \(track.writer.url.path) (\(clock(track.writer.durationSeconds)))")
        }
        line("  Log:     \(session.logURL.path)")
        line("  Session: \(session.manifestURL.path)")
    }

    private func meter(_ db: Float) -> String {
        let width = 10
        let filled = Int(((db + 60) / 60 * Float(width)).rounded())
        let bar = String(repeating: "#", count: max(0, min(width, filled)))
        return "[" + bar.padding(toLength: width, withPad: " ", startingAt: 0) + "]"
    }

    private func clock(_ seconds: Double) -> String {
        let total = Int(seconds)
        return String(format: "%02d:%02d:%02d", total / 3600, total / 60 % 60, total % 60)
    }

    private func clearStatus() {
        if statusVisible { write("\r\u{1B}[2K") }
        statusVisible = false
    }

    private func line(_ text: String) { write(text + "\n") }

    private func write(_ text: String) {
        FileHandle.standardOutput.write(Data(text.utf8))
    }
}
