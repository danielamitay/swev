import Foundation
import Swev

/// Downloads a compatible package once and runs a text or single-image decision.
@main
struct Decisions {
    static func main() async {
        let arguments = Array(CommandLine.arguments.dropFirst())
        if arguments == ["--help"] {
            print("Usage: Decisions OWNER/REPO MODEL.mlpackage [IMAGE.png|IMAGE.jpg]")
            return
        }
        do {
            guard (2...3).contains(arguments.count) else {
                throw SwevError.invalidRequest("Usage: Decisions OWNER/REPO MODEL.mlpackage [IMAGE.png|IMAGE.jpg]")
            }
            FileHandle.standardError.write(Data("Loading model; the first run downloads the package before Core ML compilation.\n".utf8))
            let model = try await SwevModel.load(
                from: HuggingFaceModel(repository: arguments[0], package: arguments[1]),
                configuration: .init(computeUnits: .cpuOnly)
            )
            let response: DecisionResponse
            if arguments.count == 3 {
                guard model.descriptor.capabilities.supportsImages else { throw SwevError.unsupportedModality }
                let imageURL = URL(fileURLWithPath: arguments[2])
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
