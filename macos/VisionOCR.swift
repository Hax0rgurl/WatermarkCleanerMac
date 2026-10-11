import Foundation
import Vision

// A persistent local OCR worker. It reads image data, never file paths or URLs.
func recognize(_ line: String) throws -> [[String: Any]] {
    guard let json = line.data(using: .utf8),
          let input = try JSONSerialization.jsonObject(with: json) as? [String: Any],
          let encoded = input["image"] as? String,
          let image = Data(base64Encoded: encoded) else {
        throw NSError(domain: "WatermarkOCR", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "Invalid image request"])
    }
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false
    request.minimumTextHeight = 0.006
    request.usesCPUOnly = true
    request.preferBackgroundProcessing = true
    try VNImageRequestHandler(data: image, options: [:]).perform([request])
    return (request.results ?? []).compactMap { observation in
        guard let text = observation.topCandidates(1).first else { return nil }
        let box = observation.boundingBox
        return ["text": text.string, "confidence": text.confidence,
                "bbox": [box.minX, 1 - box.maxY, box.maxX, 1 - box.minY]]
    }
}

while let line = readLine() {
    autoreleasepool {
        let result: [String: Any]
        do {
            result = ["text": try recognize(line)]
        } catch {
            result = ["error": error.localizedDescription]
        }
        if let data = try? JSONSerialization.data(withJSONObject: result, options: [.sortedKeys]),
           let output = String(data: data, encoding: .utf8) {
            print(output)
            fflush(stdout)
        }
    }
}
