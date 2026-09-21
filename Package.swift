// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "Swev",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [.library(name: "Swev", targets: ["Swev"]), .executable(name: "swev", targets: ["SwevCLI"])],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", exact: "3.31.4"),
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.4"),
        .package(url: "https://github.com/huggingface/swift-huggingface", exact: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.3.0"),
    ],
    targets: [
        .target(name: "Swev", dependencies: [
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXLLM", package: "mlx-swift-lm"),
            .product(name: "MLXVLM", package: "mlx-swift-lm"),
            .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
            .product(name: "HuggingFace", package: "swift-huggingface"),
            .product(name: "Tokenizers", package: "swift-transformers"),
        ]),
        .executableTarget(name: "SwevCLI", dependencies: ["Swev"]),
        .testTarget(name: "SwevTests", dependencies: ["Swev"]),
    ]
)
