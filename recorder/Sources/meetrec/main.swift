import Foundation

let usage = """
    Usage:
      meetrec record [--out DIR | --session-dir DIR] [--mic NAME_OR_UID] [--max-minutes N]
      meetrec devices

    record   Records the microphone and all system audio into a new session folder
             under DIR (default: ~/Meetings). Shows live levels and health warnings.
             Press Ctrl-C to stop and save. --session-dir records into exactly DIR,
             which must not exist yet.
    devices  Lists microphones and their UIDs.
    """

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("error: \(message)\n".utf8))
    exit(2)
}

func parseRecordOptions(_ arguments: ArraySlice<String>) -> RecordOptions {
    var options = RecordOptions(
        outputRoot: FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Meetings"))
    var iterator = arguments.makeIterator()
    while let argument = iterator.next() {
        func value() -> String {
            guard let next = iterator.next() else { fail("\(argument) needs a value\n\(usage)") }
            return next
        }
        switch argument {
        case "--out": options.outputRoot = URL(fileURLWithPath: (value() as NSString).expandingTildeInPath)
        case "--mic": options.micQuery = value()
        case "--session-dir":
            options.sessionDirectory = URL(fileURLWithPath: (value() as NSString).expandingTildeInPath).standardizedFileURL
        case "--max-minutes":
            let raw = value()
            guard let minutes = Double(raw), minutes > 0 else { fail("--max-minutes must be a positive number, got \(raw)") }
            options.maxMinutes = minutes
        default: fail("unknown option \(argument)\n\(usage)")
        }
    }
    return options
}

let arguments = CommandLine.arguments.dropFirst()
do {
    switch arguments.first {
    case "record":
        exit(try Recorder(options: parseRecordOptions(arguments.dropFirst())).run())
    case "devices":
        let defaultID = try? Devices.defaultInputID()
        for device in try Devices.inputDevices() {
            let marker = device.id == defaultID ? "*" : " "
            print("\(marker) \(device.name)  [\(device.uid)]  \(Int(device.sampleRate)) Hz, \(device.channels) ch")
        }
    case "-h", "--help", "help":
        print(usage)
    default:
        fail(usage)
    }
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
