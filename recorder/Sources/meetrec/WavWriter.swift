import Foundation

/// Streams 16-bit mono PCM to a WAV file and keeps the header valid after every
/// flush, so a crash or power loss leaves a playable file with all flushed audio.
final class WavWriter {
    static let maxDataBytes = 3_800_000_000

    let url: URL
    let sampleRate: Int
    private let handle: FileHandle
    private var scratch = [Int16]()
    private(set) var frames = 0

    init(url: URL, sampleRate: Int) throws {
        self.url = url
        self.sampleRate = sampleRate
        guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
            throw SetupError("Cannot create \(url.path).", fix: "Check that the folder exists and is writable.")
        }
        handle = try FileHandle(forWritingTo: url)
        try handle.write(contentsOf: Self.header(sampleRate: sampleRate, dataBytes: 0))
    }

    var isFull: Bool { frames * 2 >= Self.maxDataBytes }
    var durationSeconds: Double { Double(frames) / Double(sampleRate) }

    func append(_ samples: UnsafeBufferPointer<Float>) throws {
        if scratch.count < samples.count { scratch = [Int16](repeating: 0, count: samples.count) }
        for (i, sample) in samples.enumerated() {
            let clamped = max(-1, min(1, sample))
            scratch[i] = Int16(clamped * Float(Int16.max)).littleEndian
        }
        try scratch.withUnsafeBytes { bytes in
            try handle.write(contentsOf: UnsafeRawBufferPointer(rebasing: bytes[0..<(samples.count * 2)]))
        }
        frames += samples.count
    }

    func appendSilence(frames count: Int) throws {
        guard count > 0 else { return }
        try handle.write(contentsOf: Data(count: count * 2))
        frames += count
    }

    func updateHeader() throws {
        let dataBytes = UInt32(frames * 2)
        try handle.seek(toOffset: 4)
        try handle.write(contentsOf: Self.le(36 + dataBytes))
        try handle.seek(toOffset: 40)
        try handle.write(contentsOf: Self.le(dataBytes))
        try handle.seekToEnd()
    }

    func sync() throws { try handle.synchronize() }

    func close() throws {
        try updateHeader()
        try handle.synchronize()
        try handle.close()
    }

    private static func le(_ value: UInt32) -> Data { withUnsafeBytes(of: value.littleEndian) { Data($0) } }
    private static func le16(_ value: UInt16) -> Data { withUnsafeBytes(of: value.littleEndian) { Data($0) } }

    private static func header(sampleRate: Int, dataBytes: UInt32) -> Data {
        var data = Data("RIFF".utf8)
        data += le(36 + dataBytes)
        data += Data("WAVEfmt ".utf8)
        data += le(16)
        data += le16(1)  // PCM
        data += le16(1)  // mono
        data += le(UInt32(sampleRate))
        data += le(UInt32(sampleRate * 2))
        data += le16(2)
        data += le16(16)
        data += Data("data".utf8)
        data += le(dataBytes)
        return data
    }
}
