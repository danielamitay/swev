import Foundation
import Swev

@main
struct SwevCLI {
    static func main() async {
        do {
            let arguments = Array(CommandLine.arguments.dropFirst())
            guard [2, 4].contains(arguments.count), arguments[0] == "--model" else {
                throw SwevError.invalidRequest("Usage: swev --model MODEL_ID_OR_DIRECTORY [--max-context-tokens N] (JSON Lines on stdin)")
            }
            let contextLimit: Int?
            if arguments.count == 4 {
                guard arguments[2] == "--max-context-tokens", let value = Int(arguments[3]), value > 0 else {
                    throw SwevError.invalidRequest("--max-context-tokens requires a positive integer")
                }
                contextLimit = value
            } else { contextLimit = nil }
            let model: SwevModel
            let location = URL(fileURLWithPath: arguments[1])
            if FileManager.default.fileExists(atPath: location.path) || arguments[1].hasPrefix("/") || arguments[1].hasPrefix(".") {
                model = try await SwevModel.load(url: location, maxContextTokens: contextLimit)
            } else {
                model = try await SwevModel.load(hf: arguments[1], maxContextTokens: contextLimit)
            }
            while let line = readLine() {
                let request = try DecisionCodec.decodeRequest(Data(line.utf8), modelID: model.descriptor.id)
                let response = try await model.predict(request)
                print(String(decoding: try DecisionCodec.encodeResponse(response), as: UTF8.self))
            }
        } catch {
            FileHandle.standardError.write(Data("Swev: \(error)\n".utf8))
            exit(2)
        }
    }
}
