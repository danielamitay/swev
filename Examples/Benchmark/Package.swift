// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "SwevBenchmark",
    platforms: [.macOS(.v15)],
    dependencies: [.package(name: "Swev", path: "../..")],
    targets: [.executableTarget(name: "SwevBenchmark", dependencies: [.product(name: "Swev", package: "Swev")])]
)
