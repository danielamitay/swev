import Foundation
import Swev

@main
struct SwevCLI {
    static func main() async {
        do {
            let arguments = Array(CommandLine.arguments.dropFirst())
            guard arguments.count == 2, arguments[0] == "--model" else {
                throw SwevError.invalidRequest("Usage: swev --model PATH (JSON Lines on stdin)")
            }
            let model = try await SwevModel.load(from: URL(fileURLWithPath: arguments[1]), configuration: .init(computeUnits: .cpuOnly))
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
