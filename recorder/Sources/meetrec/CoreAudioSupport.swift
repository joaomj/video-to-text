import CoreAudio
import Foundation

struct CoreAudioError: Error, CustomStringConvertible {
    let status: OSStatus
    let operation: String

    var description: String {
        let code = UInt32(bitPattern: status)
        let bytes = [24, 16, 8, 0].map { UInt8((code >> UInt32($0)) & 0xFF) }
        let printable = bytes.allSatisfy { $0 >= 32 && $0 < 127 }
        let fourCC = printable ? " '\(String(decoding: bytes, as: UTF8.self))'" : ""
        return "\(operation) failed (OSStatus \(status)\(fourCC))"
    }
}

func check(_ status: OSStatus, _ operation: String) throws {
    if status != noErr { throw CoreAudioError(status: status, operation: operation) }
}

func propertyAddress(
    _ selector: AudioObjectPropertySelector,
    _ scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal
) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(
        mSelector: selector, mScope: scope, mElement: kAudioObjectPropertyElementMain)
}

func readValue<T: BitwiseCopyable>(
    _ object: AudioObjectID,
    _ selector: AudioObjectPropertySelector,
    scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal,
    initial: T
) throws -> T {
    var address = propertyAddress(selector, scope)
    var size = UInt32(MemoryLayout<T>.size)
    var value = initial
    try check(
        AudioObjectGetPropertyData(object, &address, 0, nil, &size, &value),
        "Reading audio property \(selector) of object \(object)")
    return value
}

func readString(
    _ object: AudioObjectID,
    _ selector: AudioObjectPropertySelector,
    scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal
) throws -> String {
    var address = propertyAddress(selector, scope)
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    var value: Unmanaged<CFString>?
    try check(
        AudioObjectGetPropertyData(object, &address, 0, nil, &size, &value),
        "Reading audio string property \(selector) of object \(object)")
    guard let value else {
        throw CoreAudioError(status: kAudioHardwareUnspecifiedError, operation: "Empty string property")
    }
    return value.takeRetainedValue() as String
}

func readObjectList(
    _ object: AudioObjectID,
    _ selector: AudioObjectPropertySelector,
    scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal
) throws -> [AudioObjectID] {
    var address = propertyAddress(selector, scope)
    var size: UInt32 = 0
    try check(
        AudioObjectGetPropertyDataSize(object, &address, 0, nil, &size),
        "Reading audio list size \(selector)")
    var ids = [AudioObjectID](
        repeating: kAudioObjectUnknown, count: Int(size) / MemoryLayout<AudioObjectID>.size)
    if ids.isEmpty { return [] }
    try check(
        AudioObjectGetPropertyData(object, &address, 0, nil, &size, &ids),
        "Reading audio list \(selector)")
    return ids
}
