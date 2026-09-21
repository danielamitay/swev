import Foundation

/// Lock-protected admission happens before the actor hop, so its mailbox stays bounded.
final class RequestAdmission: @unchecked Sendable {
    private let lock = NSLock()
    private let limit: Int
    private var pending = 0
    init(limit: Int) { self.limit = limit }
    func acquire() throws {
        lock.lock(); defer { lock.unlock() }
        guard pending < limit else { throw SwevError.queueFull }
        pending += 1
    }
    func release() {
        lock.lock(); defer { lock.unlock() }
        pending -= 1
    }
}
