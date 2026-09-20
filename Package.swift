// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Swev",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [.library(name: "Swev", targets: ["Swev"])],
    targets: [
        .target(name: "Swev"),
        .testTarget(name: "SwevTests", dependencies: ["Swev"]),
    ]
)
