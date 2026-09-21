import Foundation
import Swev

@main
struct SwevCLI {
    static func main() async {
        do {
            let arguments = Array(CommandLine.arguments.dropFirst())
            guard arguments.count == 2, arguments[0] == "--model" else {
                throw SwevError.invalidRequest("Usage: swev --model MODEL_ID_OR_DIRECTORY (JSON Lines on stdin)")
            }
            let model: SwevModel
            let location = URL(fileURLWithPath: arguments[1])
            if ["mlpackage", "mlmodel", "mlmodelc"].contains(location.pathExtension) {
                model = try await SwevModel.load(from: location, configuration: .init(computeUnits: .cpuOnly))
            } else if FileManager.default.fileExists(atPath: location.path) || arguments[1].hasPrefix("/") || arguments[1].hasPrefix(".") {
                model = try await SwevModel.load(url: location)
            } else {
                model = try await SwevModel.load(hf: arguments[1])
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
