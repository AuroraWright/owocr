import Foundation
import Cocoa
import VisionKit

class StandardError: TextOutputStream {
    func write(_ string: String) {
        try! FileHandle.standardError.write(contentsOf: Data(string.utf8))
    }
}

@main
struct Litex {
    static var stderr = StandardError()

    static func main() async throws {
        try? await recognize()
    }
    
    static func recognize() async throws {
        if #available(macOS 13.0, *) {
            if let inputData = try? FileHandle.standardInput.readToEnd() {
                if let image = NSImage(data: inputData) {
                    try? await recognizeTextByLiveText(inputimage: image)
                }
            }
        }
    }

    @available(macOS 13, *)
    static func recognizeTextByLiveText(inputimage: NSImage) async throws {
        // if not supported
        if !ImageAnalyzer.isSupported {
            print("Live Text is not supported", to: &stderr)
            return
        }

        // setup ImageAnalyzer
        var configuration = ImageAnalyzer.Configuration([.text])
        configuration.locales = ["ja","en"]
        let analyzer = ImageAnalyzer()
        
        // analyze the image
        let analysis = try? await analyzer.analyze(inputimage,
                                                   orientation: .up,
                                                   configuration: configuration)
        // output results
        if let analysis {
            if analysis.hasResults(for: .text) {
                print(analysis.transcript)
            } else {
                print("")
            }
        }
        else {
            print("Unknown error", to: &stderr)
        }
    }
}
