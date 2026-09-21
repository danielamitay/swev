// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "SwevExamples",
    platforms: [.macOS(.v15)],
    dependencies: [.package(name: "Swev", path: "../..")],
    targets: [
        .executableTarget(name: "Decisions", dependencies: [.product(name: "Swev", package: "Swev")]),
    ]
)
