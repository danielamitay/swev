import Foundation
import Hub
import MLXHuggingFace
import MLXLMCommon
import Tokenizers

/// Keeps upstream tokenizer/template loading while bridging a Python-only role string method.
/// Checkpoint files are unchanged; tokenization and template evaluation remain runtime-owned.
struct TokenizerCompatibility: MLXLMCommon.TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let source = LanguageModelConfigurationFromHub(modelFolder: directory)
        guard let original = try await source.tokenizerConfig else {
            throw SwevError.invalidModelAsset
        }
        var configuration = original
        if let template = original.chatTemplate.string(), var fields = original.dictionary() {
            fields["chat_template"] = Config(Self.portableTemplate(template))
            configuration = Config(fields)
        }
        let tokenizer = try PreTrainedTokenizer(tokenizerConfig: configuration,
            tokenizerData: try await source.tokenizerData, strict: true)
        return #adaptHuggingFaceTokenizer(tokenizer)
    }

    /// Converts only a complete role-capitalization expression, not arbitrary template text.
    static func portableTemplate(_ template: String) -> String {
        let expression = #"\{\{\s*([A-Za-z_][A-Za-z_0-9]*\[['"]role['"]\])\.capitalize\(\)\s*\}\}"#
        return template.replacingOccurrences(of: expression, with: "{{ $1 | capitalize }}", options: .regularExpression)
    }
}
