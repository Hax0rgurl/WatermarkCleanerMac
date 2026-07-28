import Darwin
import Foundation

private struct FileIdentity: Codable, Equatable {
    let device: UInt64
    let inode: UInt64
    let fileType: UInt32
    let size: Int64
    let mtimeNanoseconds: Int64
}

private struct TrashResponse: Codable {
    let status: String
    let source: String
    let trashed: String?
    let device: UInt64
    let inode: UInt64
}

private func fail(_ message: String, status: Int32) -> Never {
    fputs("TRASH_ERROR \(message)\n", stderr)
    exit(status)
}

private func identity(at path: String) -> FileIdentity? {
    var details = stat()
    let status = path.withCString { pointer in
        lstat(pointer, &details)
    }
    guard status == 0 else { return nil }
    let mtimeNanoseconds =
        Int64(details.st_mtimespec.tv_sec) * 1_000_000_000
        + Int64(details.st_mtimespec.tv_nsec)
    return FileIdentity(
        device: UInt64(details.st_dev),
        inode: UInt64(details.st_ino),
        fileType: UInt32(details.st_mode & S_IFMT),
        size: Int64(details.st_size),
        mtimeNanoseconds: mtimeNanoseconds
    )
}

private func encodedReceipt(_ response: TrashResponse) -> String? {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    guard let payload = try? encoder.encode(response) else { return nil }
    return String(data: payload, encoding: .utf8)
}

private func emitReceipt(
    _ response: TrashResponse,
    fallbackUnknownReceipt: String
) -> Never {
    // The fallback is fully encoded before Foundation is called, so even an
    // impossible post-commit serialization failure remains explicit JSON with
    // status=location_unknown rather than a misleading process failure.
    print(encodedReceipt(response) ?? fallbackUnknownReceipt)
    exit(0)
}

guard CommandLine.arguments.count == 7 else {
    fail(
        "usage: trash-item PATH DEVICE INODE FILE_TYPE SIZE MTIME_NS",
        status: 64
    )
}

let source = URL(
    fileURLWithPath: CommandLine.arguments[1]
).standardizedFileURL
guard source.path.hasPrefix("/") else {
    fail("the source path must be absolute", status: 64)
}
guard let device = UInt64(CommandLine.arguments[2]),
      let inode = UInt64(CommandLine.arguments[3]),
      let fileType = UInt32(CommandLine.arguments[4]),
      let size = Int64(CommandLine.arguments[5]),
      let mtimeNanoseconds = Int64(CommandLine.arguments[6]) else {
    fail("the expected file identity is invalid", status: 64)
}

private let expected = FileIdentity(
    device: device,
    inode: inode,
    fileType: fileType,
    size: size,
    mtimeNanoseconds: mtimeNanoseconds
)
guard expected.fileType == UInt32(S_IFREG) else {
    fail("only a selected regular file may be moved to Trash", status: 65)
}
guard let current = identity(at: source.path) else {
    fail("the selected path is unavailable", status: 66)
}
guard current.fileType == UInt32(S_IFREG) else {
    fail("the selected path is no longer a regular file", status: 65)
}
guard current == expected else {
    fail("the selected file identity changed; refusing to Trash it", status: 65)
}

private let unknownResponse = TrashResponse(
    status: "location_unknown",
    source: source.path,
    trashed: nil,
    device: expected.device,
    inode: expected.inode
)
guard let fallbackUnknownReceipt = encodedReceipt(unknownResponse) else {
    fail("could not prepare a safe Trash receipt", status: 70)
}

// No success-shaped JSON is emitted before Foundation returns successfully.
var trashedURL: NSURL?
do {
    try FileManager.default.trashItem(
        at: source,
        resultingItemURL: &trashedURL
    )
} catch {
    fail(error.localizedDescription, status: 74)
}

guard let trashed = trashedURL as URL? else {
    emitReceipt(
        unknownResponse,
        fallbackUnknownReceipt: fallbackUnknownReceipt
    )
}

emitReceipt(
    TrashResponse(
        status: "trashed",
        source: source.path,
        trashed: trashed.path,
        device: expected.device,
        inode: expected.inode
    ),
    fallbackUnknownReceipt: fallbackUnknownReceipt
)
