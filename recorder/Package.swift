// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "meetrec",
    platforms: [.macOS(.v15)],
    targets: [
        .executableTarget(
            name: "meetrec",
            path: "Sources/meetrec",
            swiftSettings: [.swiftLanguageMode(.v5)]
        )
    ]
)
