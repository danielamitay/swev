import Foundation
import Swev

/// Loads an ordinary MLX model once and runs a text or single-image decision.
@main
struct Decisions {
    static func main() async {
        var arguments = Array(CommandLine.arguments.dropFirst())
        let usage = "Usage: Decisions MODEL_ID_OR_DIRECTORY [IMAGE.png|IMAGE.jpg] [--max-context-tokens N]"
        if arguments == ["--help"] {
            print(usage)
            return
        }
        do {
            var contextLimit: Int?
            if arguments.count >= 2, arguments[arguments.count - 2] == "--max-context-tokens" {
                guard let value = Int(arguments.last!), value > 0 else {
                    throw SwevError.invalidRequest("--max-context-tokens requires a positive integer")
                }
                contextLimit = value
                arguments.removeLast(2)
            }
            guard (1...2).contains(arguments.count) else {
                throw SwevError.invalidRequest(usage)
            }
            FileHandle.standardError.write(Data("Loading model; the first Hub load downloads its weights.\n".utf8))
            let location = URL(fileURLWithPath: arguments[0])
            let model: SwevModel
            if FileManager.default.fileExists(atPath: location.path) || arguments[0].hasPrefix("/") || arguments[0].hasPrefix(".") {
                model = try await SwevModel.load(url: location, maxContextTokens: contextLimit)
            } else {
                model = try await SwevModel.load(hf: arguments[0], maxContextTokens: contextLimit)
            }
            let response: DecisionResponse
            if arguments.count == 2 {
                guard model.descriptor.capabilities.supportsImages else { throw SwevError.unsupportedModality }
                let imageURL = URL(fileURLWithPath: arguments[1])
                let contentType: String
                switch imageURL.pathExtension.lowercased() {
                case "png": contentType = "image/png"
                case "jpg", "jpeg": contentType = "image/jpeg"
                default: throw SwevError.invalidRequest("Use a PNG or JPEG image")
                }
                response = try await model.predict(
                    state: "Look at the attached image.",
                    questions: [.choice(id: "scene", instructions: "Where was this photo taken?",
                                        options: [.init(id: "indoors"), .init(id: "outdoors")])],
                    images: [.init(data: try Data(contentsOf: imageURL), contentType: contentType)]
                )
            } else {
                response = try await model.predict(
                    state: "Checkout is failing for every customer. Please investigate immediately.",
                    questions: [
                        .choice(id: "route", instructions: "Which team should handle this issue?",
                                options: [.init(id: "engineering"), .init(id: "sales"), .init(id: "other")]),
                        .score(id: "urgency", instructions: "How urgent is this issue?", levels: ["low", "medium", "high"]),
                        .noul(id: "actionable", instructions: "Does this message request action?")
                    ]
                )
            }
            print(String(decoding: try DecisionCodec.encodeResponse(response), as: UTF8.self))
        } catch {
            FileHandle.standardError.write(Data("Decisions: \(error)\n".utf8))
            exit(2)
        }
    }
}
