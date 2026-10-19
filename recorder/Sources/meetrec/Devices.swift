import CoreAudio
import Foundation

struct InputDevice {
    let id: AudioObjectID
    let uid: String
    let name: String
    let channels: Int
    let sampleRate: Double
}

enum Devices {
    static func defaultInputID() throws -> AudioObjectID {
        let id = try readValue(
            AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyDefaultInputDevice,
            initial: AudioObjectID(kAudioObjectUnknown))
        if id == kAudioObjectUnknown {
            throw SetupError(
                "No default microphone is configured.",
                fix: "Connect a microphone or pick one in System Settings > Sound > Input.")
        }
        return id
    }

    static func inputChannelCount(_ id: AudioObjectID) throws -> Int {
        var address = propertyAddress(kAudioDevicePropertyStreamConfiguration, kAudioObjectPropertyScopeInput)
        var size: UInt32 = 0
        try check(
            AudioObjectGetPropertyDataSize(id, &address, 0, nil, &size), "Reading input stream layout size")
        if size == 0 { return 0 }
        let raw = UnsafeMutableRawPointer.allocate(byteCount: Int(size), alignment: 16)
        defer { raw.deallocate() }
        try check(AudioObjectGetPropertyData(id, &address, 0, nil, &size, raw), "Reading input stream layout")
        let list = UnsafeMutableAudioBufferListPointer(raw.assumingMemoryBound(to: AudioBufferList.self))
        return list.reduce(0) { $0 + Int($1.mNumberChannels) }
    }

    static func describe(_ id: AudioObjectID) throws -> InputDevice {
        InputDevice(
            id: id,
            uid: try readString(id, kAudioDevicePropertyDeviceUID),
            name: try readString(id, kAudioObjectPropertyName),
            channels: try inputChannelCount(id),
            sampleRate: try nominalSampleRate(id))
    }

    static func inputDevices() throws -> [InputDevice] {
        try readObjectList(AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyDevices)
            .map(describe)
            .filter { $0.channels > 0 }
    }

    /// Resolves `query` (exact UID, or case-insensitive name substring) or the default input.
    static func resolveInput(_ query: String?) throws -> InputDevice {
        guard let query else { return try describe(defaultInputID()) }
        let devices = try inputDevices()
        if let exact = devices.first(where: { $0.uid == query }) { return exact }
        let matches = devices.filter { $0.name.localizedCaseInsensitiveContains(query) }
        if matches.count == 1 { return matches[0] }
        let names = devices.map { "\"\($0.name)\"" }.joined(separator: ", ")
        if matches.isEmpty {
            throw SetupError("No microphone matches \"\(query)\".", fix: "Available inputs: \(names).")
        }
        throw SetupError(
            "\"\(query)\" matches several microphones.", fix: "Use a longer name or the UID from `meetrec devices`.")
    }

    static func nominalSampleRate(_ id: AudioObjectID) throws -> Double {
        try readValue(id, kAudioDevicePropertyNominalSampleRate, initial: Float64(0))
    }

    /// True while any process is playing through the default output device.
    static func defaultOutputIsPlaying() -> Bool {
        guard let output = try? readValue(
            AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyDefaultOutputDevice,
            initial: AudioObjectID(kAudioObjectUnknown)), output != kAudioObjectUnknown
        else { return false }
        return (try? readValue(output, kAudioDevicePropertyDeviceIsRunningSomewhere, initial: UInt32(0))) == 1
    }

    static func isAlive(_ id: AudioObjectID) -> Bool {
        (try? readValue(id, kAudioDevicePropertyDeviceIsAlive, initial: UInt32(0))) == 1
    }

    static func inputStreamFormat(_ id: AudioObjectID) throws -> AudioStreamBasicDescription {
        let streams = try readObjectList(id, kAudioDevicePropertyStreams, scope: kAudioObjectPropertyScopeInput)
        guard let first = streams.first else {
            throw SetupError("The selected device has no input stream.", fix: "Pick another microphone with --mic.")
        }
        return try readValue(first, kAudioStreamPropertyVirtualFormat, initial: AudioStreamBasicDescription())
    }
}
