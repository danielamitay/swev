import CryptoKit
import Foundation

/// A self-contained .mlpackage directory in a Hugging Face model repository.
public struct HuggingFaceModel: Sendable {
    public enum CachePolicy: Sendable {
        /// Reuse a complete local download without contacting the Hub.
        case useCache
        /// Resolve the revision again; download only if its commit changed.
        case refresh
        /// Never access the network; throw if the package is not cached.
        case localOnly
    }

    /// Hub model repository in `owner/name` form.
    public var repository: String
    /// Exact package directory within the repository, including `.mlpackage`.
    public var package: String
    /// Branch, tag, or commit SHA. Pin a full SHA for reproducible downloads.
    public var revision: String
    /// Optional read token for private/gated repositories; never persisted by Swev.
    public var token: String?
    /// Root for downloaded snapshots; defaults to `Swev/HuggingFace` in the user's caches directory.
    public var cacheDirectory: URL
    public var cachePolicy: CachePolicy

    public init(repository: String, package: String, revision: String = "main", token: String? = nil,
                cacheDirectory: URL? = nil, cachePolicy: CachePolicy = .useCache) {
        self.repository = repository
        self.package = package
        self.revision = revision
        self.token = token
        self.cacheDirectory = cacheDirectory ?? FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Swev/HuggingFace", isDirectory: true)
        self.cachePolicy = cachePolicy
    }

    /// Returns a cached package or downloads a complete, commit-pinned snapshot atomically.
    /// The default policy reuses cached revisions without a network request. Compiled models are not cached.
    /// Returned packages should be treated as read-only.
    /// - Throws: `HuggingFaceError`, network/filesystem errors, or `CancellationError`.
    public func download() async throws -> URL {
        try await download(using: HubTransport.live)
    }

    func download(using transport: HubTransport) async throws -> URL {
        try Task.checkCancellation()
        guard Self.safePath(repository), repository.split(separator: "/").count == 2,
              Self.safePath(package), package.hasSuffix(".mlpackage"),
              !revision.isEmpty, revision.utf8.count <= 256, cacheDirectory.isFileURL else {
            throw HuggingFaceError.invalidLocation
        }
        let fm = FileManager.default
        let key = SHA256.hash(data: Data("\(repository)\n\(revision)\n\(package)".utf8)).map { String(format: "%02x", $0) }.joined()
        let reference = cacheDirectory.appendingPathComponent(key, isDirectory: true)
        let receiptURL = reference.appendingPathComponent("receipt.json")
        let cached = (try? Data(contentsOf: receiptURL)).flatMap { try? JSONDecoder().decode(HubReceipt.self, from: $0) }
        func valid(_ receipt: HubReceipt) -> URL? {
            guard receipt.commit.count == 40, receipt.commit.allSatisfy(\.isHexDigit), !receipt.files.isEmpty else { return nil }
            let root = reference.appendingPathComponent(receipt.commit).appendingPathComponent("model.mlpackage")
            guard receipt.files.allSatisfy({ file in
                Self.safePath(file.path) && (try? root.appendingPathComponent(file.path).resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]))
                    .map { $0.isRegularFile == true && $0.fileSize == file.size } == true
            }) else { return nil }
            return root
        }
        if cachePolicy != .refresh, let cached, let url = valid(cached) { return url }
        if cachePolicy == .localOnly { throw HuggingFaceError.cacheMiss }
        func request(_ path: String, query: String = "") -> URLRequest {
            var request = URLRequest(url: URL(string: "https://huggingface.co/" + path + query)!)
            request.setValue("swev", forHTTPHeaderField: "User-Agent")
            if let token { request.setValue("Bearer " + token, forHTTPHeaderField: "Authorization") }
            return request
        }
        let infoRequest = request("api/models/\(Self.encodedPath(repository))/revision/\(Self.encoded(revision))", query: "?blobs=true")
        let infoData = try await transport.data(infoRequest)
        guard infoData.count <= 16 * 1024 * 1024,
              let info = try? JSONDecoder().decode(HubInfo.self, from: infoData),
              info.sha.count == 40, info.sha.allSatisfy(\.isHexDigit) else { throw HuggingFaceError.invalidResponse }
        if let cached, cached.commit == info.sha, let url = valid(cached) { return url }
        let prefix = package + "/"
        let files = info.siblings.filter { $0.rfilename.hasPrefix(prefix) }
        guard !files.isEmpty, files.count <= 1024 else { throw HuggingFaceError.packageNotFound }
        let entries = try files.map { file -> HubReceipt.File in
            let path = String(file.rfilename.dropFirst(prefix.count))
            guard Self.safePath(path), let size = file.size, size >= 0 else { throw HuggingFaceError.invalidResponse }
            return .init(path: path, size: size)
        }
        guard entries.contains(where: { $0.path == "Manifest.json" }), Set(entries.map(\.path)).count == entries.count else {
            throw HuggingFaceError.invalidResponse
        }
        try fm.createDirectory(at: reference, withIntermediateDirectories: true)
        let staging = reference.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let root = staging.appendingPathComponent("model.mlpackage", isDirectory: true)
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: staging) }
        for (file, entry) in zip(files, entries) {
            try Task.checkCancellation()
            let temporary = try await transport.file(request("\(Self.encodedPath(repository))/resolve/\(info.sha)/\(Self.encodedPath(file.rfilename))"))
            defer { try? fm.removeItem(at: temporary) }
            guard try temporary.resourceValues(forKeys: [.fileSizeKey]).fileSize == entry.size else { throw HuggingFaceError.integrityFailure }
            // Large files are hashed incrementally, never read into memory as one Data value.
            if let expected = file.lfs?.sha256 {
                let handle = try FileHandle(forReadingFrom: temporary)
                defer { try? handle.close() }
                var hash = SHA256()
                while let chunk = try handle.read(upToCount: 1_048_576), !chunk.isEmpty {
                    try Task.checkCancellation()
                    hash.update(data: chunk)
                }
                guard hash.finalize().map({ String(format: "%02x", $0) }).joined() == expected else { throw HuggingFaceError.integrityFailure }
            }
            let destination = root.appendingPathComponent(entry.path)
            try fm.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            try fm.moveItem(at: temporary, to: destination)
        }
        try Task.checkCancellation()
        let receipt = HubReceipt(commit: info.sha, files: entries)
        let snapshot = reference.appendingPathComponent(info.sha, isDirectory: true)
        if fm.fileExists(atPath: snapshot.path), valid(receipt) == nil { try fm.removeItem(at: snapshot) }
        do { try fm.moveItem(at: staging, to: snapshot) }
        catch {
            // Concurrent callers may have completed the same immutable snapshot.
            guard valid(receipt) != nil else { throw error }
        }
        try JSONEncoder().encode(receipt).write(to: receiptURL, options: .atomic)
        return snapshot.appendingPathComponent("model.mlpackage")
    }

    private static func safePath(_ value: String) -> Bool {
        !value.isEmpty && !value.contains("\\") && value.rangeOfCharacter(from: .controlCharacters) == nil &&
        value.split(separator: "/", omittingEmptySubsequences: false).allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
    }
    private static func encoded(_ value: String) -> String {
        value.addingPercentEncoding(withAllowedCharacters: CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~"))!
    }
    private static func encodedPath(_ value: String) -> String { value.split(separator: "/").map { encoded(String($0)) }.joined(separator: "/") }
}

/// Hub location, availability, response, and download-integrity failures.
public enum HuggingFaceError: Error, Sendable, Equatable {
    case invalidLocation, invalidResponse, packageNotFound, cacheMiss, integrityFailure
    case httpStatus(Int)
}

extension SwevModel {
    /// Downloads or reuses a Hub package, then loads it with the same validation as a local asset.
    /// Download caching is controlled by `source.cachePolicy`; retain the model to reuse its runtime.
    public static func load(from source: HuggingFaceModel, configuration: RuntimeConfiguration = .init()) async throws -> SwevModel {
        try await load(from: source.download(), configuration: configuration)
    }
}

private struct HubInfo: Decodable {
    let sha: String
    let siblings: [File]
    struct File: Decodable {
        let rfilename: String
        let size: Int?
        let lfs: LFS?
        struct LFS: Decodable { let sha256: String }
    }
}
private struct HubReceipt: Codable {
    let commit: String
    let files: [File]
    struct File: Codable { let path: String; let size: Int }
}

struct HubTransport: Sendable {
    var data: @Sendable (URLRequest) async throws -> Data
    var file: @Sendable (URLRequest) async throws -> URL

    static let live: Self = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 120
        configuration.timeoutIntervalForResource = 24 * 60 * 60
        let session = URLSession(configuration: configuration, delegate: HubRedirects(), delegateQueue: nil)
        @Sendable func check(_ response: URLResponse) throws {
            guard let response = response as? HTTPURLResponse else { throw HuggingFaceError.invalidResponse }
            guard response.statusCode == 200 else { throw HuggingFaceError.httpStatus(response.statusCode) }
        }
        return Self(data: { request in
            let (data, response) = try await session.data(for: request)
            try check(response)
            return data
        }, file: { request in
            let (url, response) = try await session.download(for: request)
            do { try check(response); return url }
            catch { try? FileManager.default.removeItem(at: url); throw error }
        })
    }()
}

private final class HubRedirects: NSObject, URLSessionTaskDelegate, Sendable {
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping @Sendable (URLRequest?) -> Void) {
        guard request.url?.scheme == "https" else { completionHandler(nil); return }
        var request = request
        if request.url?.host != "huggingface.co" { request.setValue(nil, forHTTPHeaderField: "Authorization") }
        completionHandler(request)
    }
}
