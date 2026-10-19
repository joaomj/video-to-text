import AVFoundation
import Foundation

enum Permissions {
    /// Requests microphone access. The terminal app that runs meetrec owns the grant.
    static func requireMicrophone() throws {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            return
        case .notDetermined:
            let semaphore = DispatchSemaphore(value: 0)
            var granted = false
            AVCaptureDevice.requestAccess(for: .audio) { granted = $0; semaphore.signal() }
            semaphore.wait()
            if granted { return }
            fallthrough
        default:
            throw SetupError(
                "Microphone access is denied for this terminal.",
                fix: "Enable your terminal in System Settings > Privacy & Security > Microphone, then restart the terminal.")
        }
    }

    /// Without this grant a Core Audio tap does not fail: it delivers digital silence.
    /// Apple exposes no public preflight, so this uses the same TCC calls as Apple's
    /// AudioCap sample. When those calls are unavailable, the check is skipped and
    /// reported as unverified.
    static func requireSystemAudio() throws -> Bool {
        typealias Preflight = @convention(c) (CFString, CFDictionary?) -> Int
        typealias Request = @convention(c) (CFString, CFDictionary?, @escaping (Bool) -> Void) -> Void
        let service = "kTCCServiceAudioCapture" as CFString
        guard let handle = dlopen("/System/Library/PrivateFrameworks/TCC.framework/Versions/A/TCC", RTLD_NOW),
              let preflightSymbol = dlsym(handle, "TCCAccessPreflight")
        else { return false }
        let preflight = unsafeBitCast(preflightSymbol, to: Preflight.self)
        if preflight(service, nil) == 0 { return true }

        if let requestSymbol = dlsym(handle, "TCCAccessRequest") {
            let request = unsafeBitCast(requestSymbol, to: Request.self)
            let semaphore = DispatchSemaphore(value: 0)
            request(service, nil) { _ in semaphore.signal() }
            semaphore.wait()
            if preflight(service, nil) == 0 { return true }
        }
        throw SetupError(
            "System audio recording is not allowed for this terminal, so remote participants would be silent.",
            fix: "Enable your terminal in System Settings > Privacy & Security > Screen & System Audio Recording (System Audio Recording Only), then restart the terminal.")
    }
}
