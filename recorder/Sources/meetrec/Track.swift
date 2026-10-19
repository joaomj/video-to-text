import CoreAudio
import Foundation

/// One recorded source: capture callback, ring buffer, WAV writer, and health monitor.
final class Track {
    let name: String
    let deviceName: String
    let sampleRate: Double
    let capture: DeviceCapture
    let health: SourceHealth
    private(set) var writer: WavWriter
    private(set) var failed = false
    private var scratch: UnsafeMutablePointer<Float>
    private let scratchCount: Int
    private var lastDelivered = 0
    private var lastDropped = 0
    /// Whether the source should be delivering frames right now. A stall is reported
    /// only when this is true and nothing arrives.
    let expectsFrames: () -> Bool

    init(name: String, deviceName: String, deviceID: AudioObjectID, sampleRate: Double, file: URL, checksFlatLine: Bool,
         expectsFrames: @escaping () -> Bool = { true }) throws {
        self.expectsFrames = expectsFrames
        self.name = name
        self.deviceName = deviceName
        self.sampleRate = sampleRate
        capture = DeviceCapture(label: name, deviceID: deviceID, ringSeconds: 20, sampleRate: sampleRate)
        writer = try WavWriter(url: file, sampleRate: Int(sampleRate.rounded()))
        health = SourceHealth(source: name, checksFlatLine: checksFlatLine, sampleRate: { sampleRate })
        scratchCount = Int(sampleRate)
        scratch = .allocate(capacity: scratchCount)
    }

    deinit { scratch.deallocate() }

    var deliveredFrames: Int { capture.ring.deliveredFrames.load(ordering: .relaxed) }
    var firstHostTime: UInt64 { capture.ring.firstHostTime.load(ordering: .relaxed) }

    /// Drains captured audio to disk and returns health transitions. Runs off the audio thread.
    ///
    /// Both files follow the wall clock from `start`. While a source sends nothing
    /// (macOS stops the system tap when nothing plays), silence fills the gap in real
    /// time, so the tracks stay aligned and health checks see the quiet period.
    func pump(elapsed: Double, since start: Date) -> [HealthEvent] {
        var events: [HealthEvent] = []
        let delivered = deliveredFrames
        let received = delivered != lastDelivered
        lastDelivered = delivered

        let dropped = capture.ring.droppedFrames.load(ordering: .relaxed)
        let newlyDropped = dropped - lastDropped
        lastDropped = dropped

        let expected = Int(Date().timeIntervalSince(start) * sampleRate)
        let pending = capture.ring.pendingFrames
        if received {
            // Close any gap left by an idle period, up to where the new audio begins.
            events += padSilence(frames: expected - pending - writer.frames - newlyDropped)
        } else {
            // Keep a margin so audio still in flight is not overlapped by padding.
            events += padSilence(frames: expected - Int(Self.idleMarginSeconds * sampleRate) - writer.frames)
        }

        while !failed {
            let count = capture.ring.consume(into: scratch, maxCount: scratchCount)
            if count == 0 { break }
            let samples = UnsafeBufferPointer(start: scratch, count: count)
            do { try writer.append(samples) } catch {
                failed = true
                events += health.reportWriteFailure(error)
            }
            events += health.accumulate(samples)
        }
        if newlyDropped > 0 { events += padSilence(frames: newlyDropped) }

        events += health.observeDelivery(
            receivedFrames: received || !expectsFrames(), elapsed: elapsed,
            alive: Devices.isAlive(capture.deviceID), droppedFrames: newlyDropped)
        return events
    }

    private static let idleMarginSeconds = 0.5

    func flush() -> [HealthEvent] {
        guard !failed else { return [] }
        do {
            try writer.updateHeader()
            try writer.sync()
            return []
        } catch {
            failed = true
            return health.reportWriteFailure(error)
        }
    }

    func finish(since start: Date) throws {
        capture.stop()
        _ = pump(elapsed: 0, since: start)
        try writer.close()
    }

    /// Writes digital silence and feeds it to the health monitor as real samples.
    private func padSilence(frames: Int) -> [HealthEvent] {
        guard !failed, frames > 0 else { return [] }
        var events: [HealthEvent] = []
        do {
            try writer.appendSilence(frames: frames)
        } catch {
            failed = true
            return health.reportWriteFailure(error)
        }
        var remaining = frames
        let zeros = UnsafeMutablePointer<Float>.allocate(capacity: scratchCount)
        defer { zeros.deallocate() }
        zeros.initialize(repeating: 0, count: scratchCount)
        while remaining > 0 {
            let chunk = min(remaining, scratchCount)
            events += health.accumulate(UnsafeBufferPointer(start: zeros, count: chunk))
            remaining -= chunk
        }
        return events
    }
}
