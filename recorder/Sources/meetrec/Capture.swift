import CoreAudio
import Foundation

/// Runs an IO callback on one HAL device and pushes a mono downmix into a ring.
final class DeviceCapture {
    let deviceID: AudioObjectID
    let ring: SampleRing
    private let label: String
    private var procID: AudioDeviceIOProcID?

    init(label: String, deviceID: AudioObjectID, ringSeconds: Double, sampleRate: Double) {
        self.label = label
        self.deviceID = deviceID
        ring = SampleRing(capacity: Int(ringSeconds * max(sampleRate, 48_000)))
    }

    func start() throws {
        let format = try Devices.inputStreamFormat(deviceID)
        let isFloat32 = format.mFormatFlags & kAudioFormatFlagIsFloat != 0 && format.mBitsPerChannel == 32
        guard format.mFormatID == kAudioFormatLinearPCM && isFloat32 else {
            throw SetupError(
                "The \(label) device delivers an unsupported sample format (\(format.mBitsPerChannel)-bit, flags \(format.mFormatFlags)).",
                fix: "Pick another input device.")
        }
        let ring = self.ring
        try check(
            AudioDeviceCreateIOProcIDWithBlock(&procID, deviceID, nil) { _, input, inputTime, _, _ in
                DeviceCapture.downmix(input, hostTime: inputTime.pointee.mHostTime, into: ring)
            }, "Creating the \(label) audio callback")
        try check(AudioDeviceStart(deviceID, procID), "Starting \(label) capture")
    }

    func stop() {
        guard let procID else { return }
        AudioDeviceStop(deviceID, procID)
        AudioDeviceDestroyIOProcID(deviceID, procID)
        self.procID = nil
    }

    /// Averages every input channel into one mono stream. Real-time safe.
    private static func downmix(
        _ input: UnsafePointer<AudioBufferList>, hostTime: UInt64, into ring: SampleRing
    ) {
        let buffers = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: input))
        guard buffers.count > 0, let firstData = buffers[0].mData else { return }
        let firstChannels = max(1, Int(buffers[0].mNumberChannels))
        let frames = Int(buffers[0].mDataByteSize) / (MemoryLayout<Float>.size * firstChannels)
        guard frames > 0 else { return }

        if buffers.count == 1 {
            let samples = firstData.assumingMemoryBound(to: Float.self)
            if firstChannels == 1 {
                ring.produce(count: frames, hostTime: hostTime) { samples[$0] }
                return
            }
            let scale = 1 / Float(firstChannels)
            ring.produce(count: frames, hostTime: hostTime) { frame in
                var sum: Float = 0
                for channel in 0..<firstChannels { sum += samples[frame * firstChannels + channel] }
                return sum * scale
            }
            return
        }

        // Non-interleaved: one buffer per channel.
        let scale = 1 / Float(buffers.count)
        ring.produce(count: frames, hostTime: hostTime) { frame in
            var sum: Float = 0
            for buffer in buffers {
                if let data = buffer.mData { sum += data.assumingMemoryBound(to: Float.self)[frame] }
            }
            return sum * scale
        }
    }
}

/// A private, unmuted global Core Audio tap wrapped in a private aggregate device.
/// It copies everything the Mac plays, whichever output device is in use, without
/// changing what the user hears.
final class SystemAudioTap {
    let tapID: AudioObjectID
    let aggregateID: AudioObjectID

    init() throws {
        let description = CATapDescription(monoGlobalTapButExcludeProcesses: [])
        description.name = "meetrec system audio"
        description.isPrivate = true
        description.muteBehavior = .unmuted

        var tap = AudioObjectID(kAudioObjectUnknown)
        let tapStatus = AudioHardwareCreateProcessTap(description, &tap)
        guard tapStatus == noErr, tap != kAudioObjectUnknown else {
            throw SetupError(
                "Could not create the system audio tap: \(CoreAudioError(status: tapStatus, operation: "AudioHardwareCreateProcessTap")).",
                fix: "Use macOS 14.4 or later and allow system audio recording for your terminal in System Settings > Privacy & Security.")
        }

        let aggregate: [String: Any] = [
            kAudioAggregateDeviceNameKey: "meetrec system audio",
            kAudioAggregateDeviceUIDKey: "meetrec.system.\(UUID().uuidString)",
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceTapListKey: [
                [
                    kAudioSubTapUIDKey: description.uuid.uuidString,
                    kAudioSubTapDriftCompensationKey: true,
                ]
            ],
        ]
        var device = AudioObjectID(kAudioObjectUnknown)
        let aggregateStatus = AudioHardwareCreateAggregateDevice(aggregate as CFDictionary, &device)
        guard aggregateStatus == noErr else {
            AudioHardwareDestroyProcessTap(tap)
            throw CoreAudioError(status: aggregateStatus, operation: "Creating the system audio device")
        }
        tapID = tap
        aggregateID = device
    }

    func destroy() {
        AudioHardwareDestroyAggregateDevice(aggregateID)
        AudioHardwareDestroyProcessTap(tapID)
    }
}
