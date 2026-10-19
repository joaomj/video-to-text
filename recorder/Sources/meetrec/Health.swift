import Foundation

/// Capture-health thresholds. Each window is long enough that normal pauses and a
/// quiet participant do not trigger a warning on their own.
enum HealthThresholds {
    /// No frames at all from a source for this long means the capture stalled.
    static let stallSeconds = 3.0
    /// Peak-to-peak range below this (about -100 dBFS) is a flat line: a real
    /// microphone always has some noise, so this means muted hardware, a revoked
    /// permission, or a broken driver.
    static let flatRange: Float = 1e-5
    static let flatSeconds = 10
    /// A one-second block with RMS above this counts as audible activity.
    static let activityRmsDb: Float = -50
    /// No audible activity for this long triggers a "no audio" warning.
    static let inactivitySeconds = 180
    /// When there is activity but the loudest peak stays below this for the
    /// window, the source is too quiet for reliable transcription.
    static let quietPeakDb: Float = -30
    static let quietSeconds = 120
}

struct BlockStats {
    let rmsDb: Float
    let peakDb: Float
    let range: Float
}

enum HealthCondition: String {
    case stalled, disconnected, flat, inactive, quiet, dropped, writeFailed
}

struct HealthEvent {
    let source: String
    let condition: HealthCondition
    let active: Bool
    let message: String
}

/// Tracks one source independently, so a working source cannot hide a failed one.
final class SourceHealth {
    let source: String
    private let checksFlatLine: Bool
    private let sampleRate: () -> Double
    private var blocks: [BlockStats] = []
    private var sumSquares: Double = 0
    private var peak: Float = 0
    private var minimum: Float = .infinity
    private var maximum: Float = -.infinity
    private var blockFrames = 0
    private var active: Set<HealthCondition> = []
    private(set) var lastBlock: BlockStats?
    private(set) var warningCount = 0
    private var secondsWithoutFrames = 0.0

    init(source: String, checksFlatLine: Bool, sampleRate: @escaping () -> Double) {
        self.source = source
        self.checksFlatLine = checksFlatLine
        self.sampleRate = sampleRate
    }

    var hasActiveProblems: Bool { !active.isEmpty }

    func accumulate(_ samples: UnsafeBufferPointer<Float>) -> [HealthEvent] {
        var events: [HealthEvent] = []
        let blockSize = max(1, Int(sampleRate()))
        for sample in samples {
            sumSquares += Double(sample * sample)
            let magnitude = abs(sample)
            if magnitude > peak { peak = magnitude }
            if sample < minimum { minimum = sample }
            if sample > maximum { maximum = sample }
            blockFrames += 1
            if blockFrames >= blockSize { events += closeBlock() }
        }
        return events
    }

    /// Called once per tick with the elapsed time and whether any frame arrived.
    func observeDelivery(receivedFrames: Bool, elapsed: Double, alive: Bool, droppedFrames: Int) -> [HealthEvent] {
        var events: [HealthEvent] = []
        secondsWithoutFrames = receivedFrames ? 0 : secondsWithoutFrames + elapsed
        events += set(
            .disconnected, !alive,
            "The \(source) device disappeared. Reconnect it, or stop and restart with another device; audio from it is missing until then.")
        events += set(
            .stalled, alive && secondsWithoutFrames >= HealthThresholds.stallSeconds,
            "No audio frames from \(source) for \(Int(HealthThresholds.stallSeconds))+ s. The device may have been switched or reset; the recording has a gap.")
        if droppedFrames > 0 {
            let message = "Dropped \(droppedFrames) \(source) frames because writing fell behind. The disk may be slow or full."
            events.append(HealthEvent(source: source, condition: .dropped, active: true, message: message))
            warningCount += 1
        }
        return events
    }

    func reportWriteFailure(_ error: Error) -> [HealthEvent] {
        set(.writeFailed, true, "Writing \(source) audio failed: \(error). Recording of this source stopped.")
    }

    private func closeBlock() -> [HealthEvent] {
        let rms = Float((sumSquares / Double(max(blockFrames, 1))).squareRoot())
        let stats = BlockStats(rmsDb: Self.decibels(rms), peakDb: Self.decibels(peak), range: maximum - minimum)
        lastBlock = stats
        blocks.append(stats)
        let keep = max(HealthThresholds.inactivitySeconds, HealthThresholds.quietSeconds, HealthThresholds.flatSeconds)
        if blocks.count > keep { blocks.removeFirst(blocks.count - keep) }
        sumSquares = 0
        peak = 0
        minimum = .infinity
        maximum = -.infinity
        blockFrames = 0
        return evaluate()
    }

    private func evaluate() -> [HealthEvent] {
        var events: [HealthEvent] = []
        if checksFlatLine {
            let recent = blocks.suffix(HealthThresholds.flatSeconds)
            let flat = recent.count == HealthThresholds.flatSeconds
                && recent.allSatisfy { $0.range < HealthThresholds.flatRange }
            events += set(
                .flat, flat,
                "The \(source) signal has been completely flat for \(HealthThresholds.flatSeconds) s. It is muted, disconnected, or missing permission. Check the mute switch and the input device.")
        }
        let window = blocks.suffix(HealthThresholds.inactivitySeconds)
        let inactive = window.count == HealthThresholds.inactivitySeconds
            && !window.contains { $0.rmsDb > HealthThresholds.activityRmsDb }
        events += set(
            .inactive, inactive && !active.contains(.flat),
            "No audible \(source) audio for \(HealthThresholds.inactivitySeconds / 60) min. " + inactiveAdvice)
        let quietWindow = blocks.suffix(HealthThresholds.quietSeconds)
        let hasActivity = quietWindow.contains { $0.rmsDb > HealthThresholds.activityRmsDb }
        let quiet = quietWindow.count == HealthThresholds.quietSeconds && hasActivity
            && !quietWindow.contains { $0.peakDb > HealthThresholds.quietPeakDb }
        events += set(
            .quiet, quiet,
            "\(source) audio is very quiet (peaks below \(Int(HealthThresholds.quietPeakDb)) dBFS for \(HealthThresholds.quietSeconds / 60) min). Transcription may miss words. " + quietAdvice)
        return events
    }

    private var inactiveAdvice: String {
        source == "mic"
            ? "If you have been speaking, check that the right microphone is selected and not muted."
            : "If others have been speaking, check that the meeting app is playing audio on this Mac."
    }

    private var quietAdvice: String {
        source == "mic" ? "Raise the input level in System Settings > Sound > Input." : "Raise the meeting app's output volume."
    }

    private func set(_ condition: HealthCondition, _ isActive: Bool, _ message: String) -> [HealthEvent] {
        if isActive == active.contains(condition) { return [] }
        if isActive {
            active.insert(condition)
            warningCount += 1
            return [HealthEvent(source: source, condition: condition, active: true, message: message)]
        }
        active.remove(condition)
        return [HealthEvent(source: source, condition: condition, active: false, message: "\(source) recovered from: \(condition.rawValue)")]
    }

    static func decibels(_ value: Float) -> Float { value > 0 ? max(-120, 20 * log10(value)) : -120 }
}
