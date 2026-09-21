import CoreML
import Foundation

/// A second graph embedded in metadata, referencing the package's shared weight blobs.
struct BundledTextModel: Decodable {
    let specification: Data
    let metadata: [String: String]
    let weights: [String: String]

    static func load(metadata shared: [String: String], compiledURL: URL, configuration: MLModelConfiguration) throws -> (MLModel, ModelAssets, URL)? {
        guard let json = shared["swev.text-model"] else { return nil }
        try Task.checkCancellation()
        guard json.utf8.count <= 12 * 1024 * 1024,
              let definition = try? JSONDecoder().decode(Self.self, from: Data(json.utf8)),
              !definition.specification.isEmpty, definition.specification.count <= 8 * 1024 * 1024,
              Set(definition.metadata.keys) == ["swev.config", "swev.preprocessing", "swev.signatures"],
              !definition.weights.isEmpty, definition.weights.count <= 16 else { throw SwevError.invalidMetadata }
        let fm = FileManager.default
        let package = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".mlpackage")
        defer { try? fm.removeItem(at: package) }
        let data = package.appendingPathComponent("Data/com.apple.CoreML")
        try fm.createDirectory(at: data.appendingPathComponent("weights"), withIntermediateDirectories: true)
        for (reference, path) in definition.weights {
            try Task.checkCancellation()
            let parts = path.split(separator: "/", omittingEmptySubsequences: false)
            guard parts.count == 2, parts[0] == "weights", !parts[1].isEmpty,
                  parts[1] != ".", parts[1] != "..", !path.contains("\\"),
                  reference == "@model_path/" + path else { throw SwevError.invalidMetadata }
            let source = compiledURL.appendingPathComponent(path)
            let destination = data.appendingPathComponent(path)
            // Hard links avoid another weight copy; external volumes can require a copy.
            do { try fm.linkItem(at: source, to: destination) }
            catch { try fm.copyItem(at: source, to: destination) }
        }
        try definition.specification.write(to: data.appendingPathComponent("model.mlmodel"))
        let manifest: [String: Any] = [
            "fileFormatVersion": "1.0.0", "rootModelIdentifier": "model",
            "itemInfoEntries": [
                "model": ["author": "com.apple.CoreML", "name": "model.mlmodel", "path": "com.apple.CoreML/model.mlmodel", "description": "Model"],
                "weights": ["author": "com.apple.CoreML", "name": "weights", "path": "com.apple.CoreML/weights", "description": "Weights"],
            ],
        ]
        try JSONSerialization.data(withJSONObject: manifest).write(to: package.appendingPathComponent("Manifest.json"))
        let compiled = try MLModel.compileModel(at: package)
        var loaded = false
        defer { if !loaded { try? fm.removeItem(at: compiled) } }
        try Task.checkCancellation()
        let textModel = try MLModel(contentsOf: compiled, configuration: configuration)
        var metadata = shared.merging(definition.metadata) { _, alternate in alternate }
        metadata.removeValue(forKey: "swev.text-model")
        let assets = try ModelAssets(model: textModel, metadata: metadata)
        let root = try CoreMLModelDescriptor.read(metadata: shared)
        guard assets.descriptor.contractVersion == root.contractVersion,
              assets.descriptor.execution.profile == "text-decision-v1",
              assets.descriptor.id == root.id,
              assets.descriptor.modelVersion == root.modelVersion,
              assets.descriptor.revision == root.revision,
              assets.descriptor.capabilities.limits.maxQuestionsPerRequest == root.capabilities.limits.maxQuestionsPerRequest,
              assets.descriptor.capabilities.limits.maxOptionsPerQuestion == root.capabilities.limits.maxOptionsPerQuestion,
              assets.descriptor.capabilities.limits.maxSequenceTokens <= root.capabilities.limits.maxSequenceTokens else { throw SwevError.invalidMetadata }
        if root.contractVersion == "2.0" {
            guard let preJSON = shared["swev.preprocessing"],
                  let pre = try? JSONDecoder().decode(ModelAssets.Preprocessing.self, from: Data(preJSON.utf8)),
                  root.capabilities.limits.maxSequenceTokens == max(pre.sequenceLength, assets.adapter.length) else { throw SwevError.invalidMetadata }
        }
        loaded = true
        return (textModel, assets, compiled)
    }
}
