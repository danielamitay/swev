import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXVLM

/// Runtime support for an explicit packed-weight manifest. No repository identifiers are inspected.
public enum RotatedQuantization {
    public enum Failure: Error { case configuration, module(String), parameters }
    public static let modelType = "prism_hadamard_qwen35"

    public static func factory(configuration: Data) throws -> VLMModelFactory {
        _ = try manifest(configuration)
        return VLMModelFactory(typeRegistry: ModelTypeRegistry(creators: [modelType: { data in
            let (config, records) = try manifest(data)
            let model = Qwen35(try JSONDecoder().decode(Qwen35Configuration.self, from: JSONSerialization.data(withJSONObject: config)))
            let leaves = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
            var replacements: [(String, Module)] = []
            for record in records {
                let path = "language_model." + record.path
                guard let original = leaves[path] else { throw Failure.module(path) }
                let rows: Int, width: Int
                if record.embedding, let embedding = original as? Embedding {
                    (rows, width) = embedding.shape
                } else if !record.embedding, let linear = original as? Linear {
                    guard linear.bias == nil else { throw Failure.module(path) }
                    (rows, width) = linear.shape
                } else { throw Failure.module(path) }
                guard width % 128 == 0, record.block == 0 || width % record.block == 0 else { throw Failure.module(path) }
                if record.embedding {
                    replacements.append((path, RotatedEmbedding(rows: rows, width: width, block: record.block)))
                } else {
                    replacements.append((path, RotatedLinear(rows: rows, width: width, block: record.block)))
                }
            }
            model.update(modules: ModuleChildren.unflattened(replacements))
            return model
        }]), processorRegistry: VLMProcessorTypeRegistry.shared, modelRegistry: VLMRegistry.shared)
    }

    private struct Record: Decodable {
        let path: String
        let block: Int
        let embedding: Bool
        let dtype: String
    }
    private static func manifest(_ data: Data) throws -> ([String: Any], [Record]) {
        guard var object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              object["model_type"] as? String == modelType, object["schema_version"] as? Int == 2,
              object["base_model_type"] as? String == "qwen3_5",
              object["tensor_namespace"] as? String == "mlx-vlm-qwen3_5",
              object["gdn_activation_layout"] as? String == "grouped",
              (object["components"] as? [String: Bool])?["vision"] == true,
              let raw = object["modules"],
              let quantization = object["quantization"] as? [String: Any],
              quantization["bits"] as? Int == 2, quantization["group_size"] as? Int == 128,
              quantization["mode"] as? String == "affine" else { throw Failure.configuration }
        let records = try JSONDecoder().decode([Record].self, from: JSONSerialization.data(withJSONObject: raw))
        guard !records.isEmpty, Set(records.map(\.path)).count == records.count,
              records.allSatisfy({ !$0.path.isEmpty && $0.dtype == "float16" && [0, 512, 1024, 2048, 4096].contains($0.block) }) else { throw Failure.configuration }
        object["model_type"] = object["base_model_type"]
        return (object, records)
    }

    /// Validate loaded signs/storage before any inference can use a transformed layer.
    public static func validate(_ model: Module) throws {
        for (_, leaf) in model.leafModules().flattened() {
            if let layer = leaf as? RotatedLinear { try layer.validate() }
            if let layer = leaf as? RotatedEmbedding { try layer.validate() }
        }
    }

    /// Orthogonal signed block transform; inverse reverses the sign/transform order.
    public static func transform(_ input: MLXArray, block: Int, signs: MLXArray, inverse: Bool = false) -> MLXArray {
        let x = input.asType(.float32)
        let oriented = inverse ? x : x * signs
        let transformed = hadamardTransform(oriented.reshaped([-1, block]), scale: 1 / sqrt(Float(block))).reshaped(input.shape)
        return (inverse ? transformed * signs : transformed).asType(input.dtype)
    }
}

private func validatePacked(weight: MLXArray, scales: MLXArray, biases: MLXArray?, block: Int, signs: MLXArray?) throws {
    guard weight.ndim == 2, weight.dtype == .uint32, scales.shape == [weight.dim(0), weight.dim(1) / 8],
          biases?.shape == scales.shape, scales.dtype == .float16, biases?.dtype == .float16 else { throw RotatedQuantization.Failure.parameters }
    if block > 0 {
        guard let signs, signs.shape == [weight.dim(1) * 16],
              all((signs .== 1) .|| (signs .== -1)).item(Bool.self) else { throw RotatedQuantization.Failure.parameters }
    } else if signs != nil { throw RotatedQuantization.Failure.parameters }
}

// Deliberately not a QuantizedLinear subclass: runtime fusion of ordinary quantized projections
// would bypass this layer's activation transform. Quantized still prevents re-quantization.
private final class RotatedLinear: Linear, Quantized {
    let groupSize = 128, bits = 2
    let mode: QuantizationMode = .affine
    let scales: MLXArray
    let biases: MLXArray
    let signs: MLXArray?
    private let block: Int
    init(rows: Int, width: Int, block: Int) {
        self.block = block; signs = block > 0 ? ones([width], dtype: .float32) : nil
        scales = zeros([rows, width / 128], dtype: .float16)
        biases = zeros([rows, width / 128], dtype: .float16)
        super.init(weight: zeros([rows, width / 16], dtype: .uint32), bias: nil)
    }
    override var shape: (Int, Int) { (weight.dim(0), weight.dim(1) * 16) }
    override func callAsFunction(_ input: MLXArray) -> MLXArray {
        let x = input.asType(.float16)
        return quantizedMM(signs.map { RotatedQuantization.transform(x, block: block, signs: $0) } ?? x,
            weight, scales: scales, biases: biases, transpose: true, groupSize: groupSize, bits: bits)
    }
    func validate() throws { try validatePacked(weight: weight, scales: scales, biases: biases, block: block, signs: signs) }
}

private final class RotatedEmbedding: Embedding, Quantized {
    let groupSize = 128, bits = 2
    let mode: QuantizationMode = .affine
    let scales: MLXArray
    let biases: MLXArray
    let signs: MLXArray?
    private let block: Int
    init(rows: Int, width: Int, block: Int) {
        self.block = block; signs = block > 0 ? ones([width], dtype: .float32) : nil
        scales = zeros([rows, width / 128], dtype: .float16); biases = zeros([rows, width / 128], dtype: .float16)
        super.init(weight: zeros([rows, width / 16], dtype: .uint32))
    }
    override var shape: (Int, Int) { (weight.dim(0), weight.dim(1) * 16) }
    override func callAsFunction(_ ids: MLXArray) -> MLXArray {
        let indices = ids.reshaped([-1])
        let x = dequantized(weight[indices], scales: scales[indices], biases: biases[indices], groupSize: groupSize, bits: bits)
            .reshaped(ids.shape + [-1]).asType(.float16)
        return signs.map { RotatedQuantization.transform(x, block: block, signs: $0, inverse: true) } ?? x
    }
    override func asLinear(_ input: MLXArray) -> MLXArray {
        let x = input.asType(.float16)
        return quantizedMM(signs.map { RotatedQuantization.transform(x, block: block, signs: $0) } ?? x,
            weight, scales: scales, biases: biases, transpose: true, groupSize: groupSize, bits: bits)
    }
    func validate() throws { try validatePacked(weight: weight, scales: scales, biases: biases, block: block, signs: signs) }
}
