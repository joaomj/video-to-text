import Synchronization

/// Single-producer, single-consumer ring of mono float samples.
///
/// The producer is the Core Audio real-time thread, so `produce` never allocates,
/// locks, or blocks. When the consumer falls behind, new samples are dropped and
/// counted so the writer can keep the timeline intact with silence.
final class SampleRing: @unchecked Sendable {
    let capacity: Int
    private let storage: UnsafeMutablePointer<Float>
    private let writeIndex = Atomic<Int>(0)
    private let readIndex = Atomic<Int>(0)
    let droppedFrames = Atomic<Int>(0)
    let deliveredFrames = Atomic<Int>(0)
    let firstHostTime = Atomic<UInt64>(0)

    init(capacity: Int) {
        self.capacity = capacity
        storage = .allocate(capacity: capacity)
        storage.initialize(repeating: 0, count: capacity)
    }

    deinit { storage.deallocate() }

    @inline(__always)
    func produce(count: Int, hostTime: UInt64, sample: (Int) -> Float) {
        if hostTime != 0 && firstHostTime.load(ordering: .relaxed) == 0 {
            firstHostTime.store(hostTime, ordering: .relaxed)
        }
        let write = writeIndex.load(ordering: .relaxed)
        let read = readIndex.load(ordering: .acquiring)
        let accepted = min(count, capacity - (write - read))
        for i in 0..<accepted {
            storage[(write + i) % capacity] = sample(i)
        }
        writeIndex.store(write + accepted, ordering: .releasing)
        if accepted < count {
            droppedFrames.wrappingAdd(count - accepted, ordering: .relaxed)
        }
        deliveredFrames.wrappingAdd(count, ordering: .relaxed)
    }

    var pendingFrames: Int {
        writeIndex.load(ordering: .acquiring) - readIndex.load(ordering: .relaxed)
    }

    func consume(into destination: UnsafeMutablePointer<Float>, maxCount: Int) -> Int {
        let read = readIndex.load(ordering: .relaxed)
        let write = writeIndex.load(ordering: .acquiring)
        let count = min(maxCount, write - read)
        for i in 0..<count {
            destination[i] = storage[(read + i) % capacity]
        }
        readIndex.store(read + count, ordering: .releasing)
        return count
    }
}
