// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Swev",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [.library(name: "Swev", targets: ["Swev"]), .executable(name: "swev", targets: ["SwevCLI"])],
    targets: [
        .target(name: "Swev"),
        .executableTarget(name: "SwevCLI", dependencies: ["Swev"]),
        .testTarget(name: "SwevTests", dependencies: ["Swev"]),
    ]
)
