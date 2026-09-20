import CryptoKit
import Foundation
import Testing
@testable import Swev

private actor HubFixture {
    var commit = String(repeating: "a", count: 40)
    var requests: [URLRequest] = []
    var corrupt = false
    var unsafe = false
    let payload = Data("example package".utf8)
    func update() { commit = String(repeating: "b", count: 40) }
    func setCorrupt() { corrupt = true }
    func setUnsafe() { unsafe = true }
    func count() -> Int { requests.count }
    func data(_ request: URLRequest) throws -> Data {
        requests.append(request)
        let hash = SHA256.hash(data: payload).map { String(format: "%02x", $0) }.joined()
        let files = ["Manifest.json", unsafe ? "../escape" : "Data/weights.bin"]
        return try JSONSerialization.data(withJSONObject: ["sha": commit, "siblings": files.map {
            ["rfilename": "example.mlpackage/" + $0, "size": payload.count, "lfs": ["sha256": hash]] as [String: Any]
        }])
    }
    func file(_ request: URLRequest) throws -> URL {
        requests.append(request)
        #expect(request.url!.path.contains("/resolve/" + commit + "/"))
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer test-token")
        let path = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try (corrupt ? Data(repeating: 0, count: payload.count) : payload).write(to: path)
        return path
    }
    nonisolated var transport: HubTransport {
        HubTransport(data: { try await self.data($0) }, file: { try await self.file($0) })
    }
}

private func hubSource() -> HuggingFaceModel {
    .init(repository: "example/model", package: "example.mlpackage", token: "test-token",
          cacheDirectory: FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString))
}

@Test func hubCachesAndRefreshesImmutableSnapshots() async throws {
    var source = hubSource()
    defer { try? FileManager.default.removeItem(at: source.cacheDirectory) }
    let hub = HubFixture()
    let first = try await source.download(using: hub.transport)
    #expect(await hub.count() == 3)
    #expect(try await source.download(using: hub.transport) == first)
    #expect(await hub.count() == 3)
    source.cachePolicy = .localOnly
    #expect(try await source.download(using: hub.transport) == first)
    source.cachePolicy = .refresh
    #expect(try await source.download(using: hub.transport) == first)
    #expect(await hub.count() == 4)
    await hub.update()
    let second = try await source.download(using: hub.transport)
    #expect(first != second)
    #expect(await hub.count() == 7)
    #expect(FileManager.default.fileExists(atPath: first.path))
}

@Test func hubRejectsCorruptDownloadsAndNeverCachesPartialPackages() async throws {
    var source = hubSource()
    defer { try? FileManager.default.removeItem(at: source.cacheDirectory) }
    let hub = HubFixture()
    await hub.setCorrupt()
    await #expect(throws: HuggingFaceError.integrityFailure) { try await source.download(using: hub.transport) }
    source.cachePolicy = .localOnly
    await #expect(throws: HuggingFaceError.cacheMiss) { try await source.download(using: hub.transport) }
}

@Test func hubRejectsTraversalAndInvalidLocations() async throws {
    var source = hubSource()
    defer { try? FileManager.default.removeItem(at: source.cacheDirectory) }
    let hub = HubFixture()
    await hub.setUnsafe()
    await #expect(throws: HuggingFaceError.invalidResponse) { try await source.download(using: hub.transport) }
    source.package = "../bad.mlpackage"
    await #expect(throws: HuggingFaceError.invalidLocation) { try await source.download(using: hub.transport) }
}

@Test func hubRepairsIncompleteCache() async throws {
    let source = hubSource()
    defer { try? FileManager.default.removeItem(at: source.cacheDirectory) }
    let hub = HubFixture()
    let path = try await source.download(using: hub.transport)
    try FileManager.default.removeItem(at: path.appendingPathComponent("Manifest.json"))
    #expect(try await source.download(using: hub.transport) == path)
    #expect(await hub.count() == 6)
}

@Test func hubCancellationAndHTTPFailureDoNotProduceCacheHits() async throws {
    let source = hubSource()
    defer { try? FileManager.default.removeItem(at: source.cacheDirectory) }
    let transport = HubTransport(data: { _ in throw HuggingFaceError.httpStatus(401) }, file: { _ in throw CancellationError() })
    await #expect(throws: HuggingFaceError.httpStatus(401)) { try await source.download(using: transport) }
    let task = Task { try await source.download(using: transport) }
    task.cancel()
    await #expect(throws: CancellationError.self) { try await task.value }
}
