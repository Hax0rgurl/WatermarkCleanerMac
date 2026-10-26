import AppKit
import Darwin
import Foundation
import UniformTypeIdentifiers
import WebKit

private let appName = "MAC WATERMARK CLEANER"
private let supportedVideoExtensions = [
    "mp4", "mov", "m4v", "avi", "mkv", "webm",
]

private struct SourceRecord: Codable, Equatable, Hashable {
    let path: String
    let device: UInt64
    let inode: UInt64
    let file_type: UInt32
    let size: Int64
    let mtime_ns: Int64

    static func capture(_ url: URL) throws -> SourceRecord {
        let normalized = url.standardizedFileURL
        var details = stat()
        let status = normalized.path.withCString { pointer in
            lstat(pointer, &details)
        }
        guard status == 0 else {
            throw NSError(
                domain: NSPOSIXErrorDomain,
                code: Int(errno),
                userInfo: [
                    NSLocalizedDescriptionKey: "The selected file is unavailable.",
                ]
            )
        }
        let fileType = UInt32(details.st_mode & S_IFMT)
        guard fileType == UInt32(S_IFREG) else {
            throw NSError(
                domain: "MACWatermarkCleaner.Selection",
                code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Only regular video files can be queued. " +
                        "Aliases, symbolic links, and folders are not accepted.",
                ]
            )
        }
        let modified =
            Int64(details.st_mtimespec.tv_sec) * 1_000_000_000
            + Int64(details.st_mtimespec.tv_nsec)
        return SourceRecord(
            path: normalized.path,
            device: UInt64(details.st_dev),
            inode: UInt64(details.st_ino),
            file_type: fileType,
            size: Int64(details.st_size),
            mtime_ns: modified
        )
    }
}

private struct SelectionManifest: Codable {
    let version: Int
    let selected_at: String
    let sources: [SourceRecord]
}

private enum Palette {
    static let void = NSColor(calibratedRed: 0.027, green: 0.024, blue: 0.067, alpha: 1)
    static let panel = NSColor(calibratedRed: 0.102, green: 0, blue: 0.2, alpha: 0.82)
    static let pink = NSColor(calibratedRed: 1, green: 0, blue: 0.431, alpha: 1)
    static let cyan = NSColor(calibratedRed: 0, green: 0.961, blue: 1, alpha: 1)
    static let violet = NSColor(calibratedRed: 0.514, green: 0.22, blue: 0.925, alpha: 1)
    static let text = NSColor(calibratedWhite: 0.97, alpha: 1)
    static let muted = NSColor(calibratedRed: 0.78, green: 0.72, blue: 0.85, alpha: 1)
    static let acid = NSColor(calibratedRed: 0.224, green: 1, blue: 0.078, alpha: 1)
}

private final class VaporwaveBackdropView: NSView {
    override var isFlipped: Bool { true }

    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        Palette.void.setFill()
        bounds.fill()

        let haze = NSGradient(colors: [
            Palette.violet.withAlphaComponent(0.32),
            Palette.pink.withAlphaComponent(0.08),
            NSColor.clear,
        ])
        haze?.draw(
            fromCenter: NSPoint(x: bounds.midX, y: -60),
            radius: 0,
            toCenter: NSPoint(x: bounds.midX, y: bounds.height * 0.7),
            radius: bounds.width * 0.7,
            options: []
        )

        let grid = NSBezierPath()
        let spacing: CGFloat = 24
        var x: CGFloat = 0
        while x <= bounds.width {
            grid.move(to: NSPoint(x: x, y: 0))
            grid.line(to: NSPoint(x: x, y: bounds.height))
            x += spacing
        }
        var y: CGFloat = 0
        while y <= bounds.height {
            grid.move(to: NSPoint(x: 0, y: y))
            grid.line(to: NSPoint(x: bounds.width, y: y))
            y += spacing
        }
        grid.lineWidth = 0.5
        Palette.cyan.withAlphaComponent(0.055).setStroke()
        grid.stroke()
    }
}

private func resourceURL() -> URL {
    let executable = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
    return executable
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("Resources", isDirectory: true)
}

private func runSelfTest() -> Never {
    let resources = resourceURL()
    let required = [
        resources.appendingPathComponent("bootstrap_macos.sh"),
        resources.appendingPathComponent("engine/app.py"),
        resources.appendingPathComponent("engine/watermark_cleaner/completion.py"),
        resources.appendingPathComponent("engine/watermark_cleaner/setup.py"),
        resources.appendingPathComponent("engine/watermark_cleaner/model_store.py"),
        resources.appendingPathComponent("engine/watermark_cleaner/engines.py"),
        resources.appendingPathComponent("engine/watermark_cleaner/backends/e2fgvi/LICENSE"),
        resources.appendingPathComponent("engine/watermark_cleaner/backends/propainter/LICENSE"),
        resources.appendingPathComponent("engine/watermark_cleaner/backends/propainter/raft/LICENSE"),
        resources.appendingPathComponent("engine/requirements-macos.lock"),
        resources.appendingPathComponent("tools/uv"),
        resources.appendingPathComponent("tools/trash-item"),
        resources.appendingPathComponent("tools/vision-ocr"),
        resources.appendingPathComponent("THIRD_PARTY_NOTICES.md"),
    ]
    for path in required where !FileManager.default.fileExists(atPath: path.path) {
        fputs("SELF_TEST_FAILED missing \(path.path)\n", stderr)
        exit(1)
    }
    for legacyPackage in ["demark_world", "sorawm"] {
        let legacy = resources.appendingPathComponent("engine/" + legacyPackage)
        if FileManager.default.fileExists(atPath: legacy.path) {
            fputs("SELF_TEST_FAILED obsolete engine package \(legacy.path)\n", stderr)
            exit(1)
        }
    }
    print("SELF_TEST_OK \(resources.path)")
    exit(0)
}

if CommandLine.arguments.contains("--self-test") {
    runSelfTest()
}

final class AppDelegate: NSObject, NSApplicationDelegate, WKNavigationDelegate, WKDownloadDelegate {
    private enum ContentMode {
        case loading
        case cleaner
        case diagnostics
    }

    private var window: NSWindow!
    private var loadingView: NSView!
    private var webView: WKWebView!
    private var diagnosticsView: NSView!
    private var statusLabel: NSTextField!
    private var selectionLabel: NSTextField!
    private var logView: NSTextView!
    private var spinner: NSProgressIndicator!
    private var diagnosticsButton: NSButton!
    private var chooseButton: NSButton!
    private var process: Process?
    private var uiURL: URL?
    private var contentMode = ContentMode.loading
    private var isTerminating = false
    private var pendingBackendOutput = ""
    private var selectionTimer: Timer?
    private var downloadDestinations: [ObjectIdentifier: URL] = [:]

    private lazy var runtimeRootURL: URL = {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(
                "Library/Application Support/MAC WATERMARK CLEANER",
                isDirectory: true
            )
    }()
    private lazy var selectionManifestURL: URL = {
        runtimeRootURL.appendingPathComponent(
            "Selections/current.json",
            isDirectory: false
        )
    }()
    private lazy var launcherLogURL: URL = {
        runtimeRootURL.appendingPathComponent("Logs/launcher.log")
    }()

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        buildWindow()
        NSApp.activate(ignoringOtherApps: true)
        window.makeKeyAndOrderFront(nil)
        startBackend()
    }

    func applicationWillTerminate(_ notification: Notification) {
        isTerminating = true
        selectionTimer?.invalidate()
        if let process, process.isRunning {
            process.terminate()
            let deadline = Date().addingTimeInterval(3)
            while process.isRunning && Date() < deadline {
                usleep(50_000)
            }
            if process.isRunning {
                Darwin.kill(process.processIdentifier, SIGKILL)
            }
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(
        _ sender: NSApplication
    ) -> Bool {
        true
    }

    private func buildWindow() {
        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1120, height: 780),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = appName
        window.titlebarAppearsTransparent = true
        window.titleVisibility = .hidden
        window.backgroundColor = Palette.void
        window.center()
        window.minSize = NSSize(width: 820, height: 600)

        let root = VaporwaveBackdropView()
        root.translatesAutoresizingMaskIntoConstraints = false
        window.contentView = root

        let header = NSView()
        header.translatesAutoresizingMaskIntoConstraints = false
        header.wantsLayer = true
        header.layer?.backgroundColor = Palette.void.withAlphaComponent(0.92).cgColor
        header.layer?.borderColor = Palette.cyan.withAlphaComponent(0.34).cgColor
        header.layer?.borderWidth = 1

        let brand = NSTextField(labelWithString: appName)
        brand.font = .monospacedSystemFont(ofSize: 16, weight: .bold)
        brand.textColor = Palette.text
        brand.translatesAutoresizingMaskIntoConstraints = false

        let brandAccent = NSTextField(labelWithString: "// PRIVATE RESTORATION")
        brandAccent.font = .monospacedSystemFont(ofSize: 9, weight: .semibold)
        brandAccent.textColor = Palette.cyan
        brandAccent.translatesAutoresizingMaskIntoConstraints = false

        selectionLabel = NSTextField(labelWithString: "NO SOURCE QUEUE")
        selectionLabel.font = .monospacedSystemFont(ofSize: 10, weight: .medium)
        selectionLabel.textColor = Palette.muted
        selectionLabel.alignment = .right
        selectionLabel.translatesAutoresizingMaskIntoConstraints = false

        chooseButton = makeButton(
            title: "CHOOSE VIDEOS",
            action: #selector(chooseVideos),
            accent: Palette.pink
        )
        let outputButton = makeButton(
            title: "OUTPUT",
            action: #selector(openOutputFolder),
            accent: Palette.cyan
        )
        diagnosticsButton = makeButton(
            title: "DIAGNOSTICS",
            action: #selector(toggleDiagnostics),
            accent: Palette.violet
        )

        let contentHost = NSView()
        contentHost.translatesAutoresizingMaskIntoConstraints = false

        loadingView = buildLoadingView()
        loadingView.translatesAutoresizingMaskIntoConstraints = false

        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        configuration.preferences.isElementFullscreenEnabled = true
        webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = self
        webView.translatesAutoresizingMaskIntoConstraints = false
        webView.isHidden = true

        diagnosticsView = buildDiagnosticsView()
        diagnosticsView.translatesAutoresizingMaskIntoConstraints = false
        diagnosticsView.isHidden = true

        root.addSubview(header)
        root.addSubview(contentHost)
        for view in [
            brand, brandAccent, selectionLabel, chooseButton,
            outputButton, diagnosticsButton,
        ].compactMap({ $0 }) {
            header.addSubview(view)
        }
        for view in [loadingView, webView, diagnosticsView].compactMap({ $0 }) {
            contentHost.addSubview(view)
        }

        NSLayoutConstraint.activate([
            header.topAnchor.constraint(equalTo: root.topAnchor),
            header.leadingAnchor.constraint(equalTo: root.leadingAnchor),
            header.trailingAnchor.constraint(equalTo: root.trailingAnchor),
            header.heightAnchor.constraint(equalToConstant: 74),

            brand.leadingAnchor.constraint(equalTo: header.leadingAnchor, constant: 24),
            brand.topAnchor.constraint(equalTo: header.topAnchor, constant: 17),
            brandAccent.leadingAnchor.constraint(equalTo: brand.leadingAnchor),
            brandAccent.topAnchor.constraint(equalTo: brand.bottomAnchor, constant: 4),

            diagnosticsButton.trailingAnchor.constraint(
                equalTo: header.trailingAnchor,
                constant: -18
            ),
            diagnosticsButton.centerYAnchor.constraint(equalTo: header.centerYAnchor),
            outputButton.trailingAnchor.constraint(
                equalTo: diagnosticsButton.leadingAnchor,
                constant: -9
            ),
            outputButton.centerYAnchor.constraint(equalTo: header.centerYAnchor),
            chooseButton.trailingAnchor.constraint(
                equalTo: outputButton.leadingAnchor,
                constant: -9
            ),
            chooseButton.centerYAnchor.constraint(equalTo: header.centerYAnchor),
            selectionLabel.trailingAnchor.constraint(
                equalTo: chooseButton.leadingAnchor,
                constant: -14
            ),
            selectionLabel.centerYAnchor.constraint(equalTo: header.centerYAnchor),
            selectionLabel.leadingAnchor.constraint(
                greaterThanOrEqualTo: brand.trailingAnchor,
                constant: 18
            ),

            contentHost.topAnchor.constraint(equalTo: header.bottomAnchor),
            contentHost.leadingAnchor.constraint(equalTo: root.leadingAnchor),
            contentHost.trailingAnchor.constraint(equalTo: root.trailingAnchor),
            contentHost.bottomAnchor.constraint(equalTo: root.bottomAnchor),

            loadingView.topAnchor.constraint(equalTo: contentHost.topAnchor),
            loadingView.leadingAnchor.constraint(equalTo: contentHost.leadingAnchor),
            loadingView.trailingAnchor.constraint(equalTo: contentHost.trailingAnchor),
            loadingView.bottomAnchor.constraint(equalTo: contentHost.bottomAnchor),

            webView.topAnchor.constraint(equalTo: contentHost.topAnchor),
            webView.leadingAnchor.constraint(equalTo: contentHost.leadingAnchor),
            webView.trailingAnchor.constraint(equalTo: contentHost.trailingAnchor),
            webView.bottomAnchor.constraint(equalTo: contentHost.bottomAnchor),

            diagnosticsView.topAnchor.constraint(equalTo: contentHost.topAnchor),
            diagnosticsView.leadingAnchor.constraint(equalTo: contentHost.leadingAnchor),
            diagnosticsView.trailingAnchor.constraint(equalTo: contentHost.trailingAnchor),
            diagnosticsView.bottomAnchor.constraint(equalTo: contentHost.bottomAnchor),
        ])

        refreshSelectionCount()
        selectionTimer = Timer.scheduledTimer(
            withTimeInterval: 1.5,
            repeats: true
        ) { [weak self] _ in
            self?.refreshSelectionCount()
        }
    }

    private func buildLoadingView() -> NSView {
        let container = NSView()
        let card = NSView()
        card.translatesAutoresizingMaskIntoConstraints = false
        card.wantsLayer = true
        card.layer?.backgroundColor = Palette.panel.cgColor
        card.layer?.borderColor = Palette.cyan.withAlphaComponent(0.75).cgColor
        card.layer?.borderWidth = 1
        card.layer?.cornerRadius = 2
        card.layer?.shadowColor = Palette.pink.cgColor
        card.layer?.shadowOpacity = 0.18
        card.layer?.shadowRadius = 24

        let kicker = NSTextField(labelWithString: "INITIALIZING // LOCAL SYSTEM")
        kicker.font = .monospacedSystemFont(ofSize: 11, weight: .bold)
        kicker.textColor = Palette.cyan
        kicker.translatesAutoresizingMaskIntoConstraints = false

        let title = NSTextField(labelWithString: "CLEAN VIDEO.\nKEEP CONTROL.")
        title.font = .monospacedSystemFont(ofSize: 34, weight: .bold)
        title.textColor = Palette.text
        title.maximumNumberOfLines = 2
        title.translatesAutoresizingMaskIntoConstraints = false

        let subtitle = NSTextField(
            wrappingLabelWithString:
                "Preparing E2FGVI-HQ and ProPainter for local processing. " +
                "The first launch downloads both models automatically."
        )
        subtitle.font = .monospacedSystemFont(ofSize: 13, weight: .regular)
        subtitle.textColor = Palette.muted
        subtitle.translatesAutoresizingMaskIntoConstraints = false

        spinner = NSProgressIndicator()
        spinner.style = .spinning
        spinner.controlSize = .regular
        spinner.startAnimation(nil)
        spinner.translatesAutoresizingMaskIntoConstraints = false

        statusLabel = NSTextField(labelWithString: "PREPARING FIRST LAUNCH…")
        statusLabel.font = .monospacedSystemFont(ofSize: 12, weight: .semibold)
        statusLabel.textColor = Palette.acid
        statusLabel.translatesAutoresizingMaskIntoConstraints = false

        let safety = NSTextField(
            wrappingLabelWithString:
                "Your videos stay on this Mac. The cleaner saves a separate verified copy " +
                "and keeps your originals."
        )
        safety.font = .monospacedSystemFont(ofSize: 11, weight: .regular)
        safety.textColor = Palette.muted
        safety.translatesAutoresizingMaskIntoConstraints = false

        container.addSubview(card)
        for view in [
            kicker, title, subtitle, spinner, statusLabel, safety,
        ].compactMap({ $0 }) {
            card.addSubview(view)
        }

        NSLayoutConstraint.activate([
            card.centerXAnchor.constraint(equalTo: container.centerXAnchor),
            card.centerYAnchor.constraint(equalTo: container.centerYAnchor, constant: -10),
            card.widthAnchor.constraint(lessThanOrEqualToConstant: 700),
            card.widthAnchor.constraint(equalTo: container.widthAnchor, multiplier: 0.72),

            kicker.topAnchor.constraint(equalTo: card.topAnchor, constant: 40),
            kicker.leadingAnchor.constraint(equalTo: card.leadingAnchor, constant: 42),
            kicker.trailingAnchor.constraint(equalTo: card.trailingAnchor, constant: -42),

            title.topAnchor.constraint(equalTo: kicker.bottomAnchor, constant: 16),
            title.leadingAnchor.constraint(equalTo: kicker.leadingAnchor),
            title.trailingAnchor.constraint(equalTo: kicker.trailingAnchor),

            subtitle.topAnchor.constraint(equalTo: title.bottomAnchor, constant: 18),
            subtitle.leadingAnchor.constraint(equalTo: kicker.leadingAnchor),
            subtitle.trailingAnchor.constraint(equalTo: kicker.trailingAnchor),

            spinner.topAnchor.constraint(equalTo: subtitle.bottomAnchor, constant: 28),
            spinner.leadingAnchor.constraint(equalTo: kicker.leadingAnchor),
            statusLabel.centerYAnchor.constraint(equalTo: spinner.centerYAnchor),
            statusLabel.leadingAnchor.constraint(equalTo: spinner.trailingAnchor, constant: 12),
            statusLabel.trailingAnchor.constraint(equalTo: kicker.trailingAnchor),

            safety.topAnchor.constraint(equalTo: spinner.bottomAnchor, constant: 30),
            safety.leadingAnchor.constraint(equalTo: kicker.leadingAnchor),
            safety.trailingAnchor.constraint(equalTo: kicker.trailingAnchor),
            safety.bottomAnchor.constraint(equalTo: card.bottomAnchor, constant: -38),
        ])
        return container
    }

    private func buildDiagnosticsView() -> NSView {
        let container = NSView()
        let title = NSTextField(labelWithString: "SYSTEM DIAGNOSTICS")
        title.font = .monospacedSystemFont(ofSize: 22, weight: .bold)
        title.textColor = Palette.text
        title.translatesAutoresizingMaskIntoConstraints = false

        let caption = NSTextField(
            wrappingLabelWithString:
                "Startup details are hidden during normal use. " +
                "They appear here when you ask for them or setup fails."
        )
        caption.font = .monospacedSystemFont(ofSize: 11, weight: .regular)
        caption.textColor = Palette.muted
        caption.translatesAutoresizingMaskIntoConstraints = false

        let scroll = NSScrollView()
        scroll.hasVerticalScroller = true
        scroll.borderType = .noBorder
        scroll.translatesAutoresizingMaskIntoConstraints = false
        scroll.wantsLayer = true
        scroll.layer?.backgroundColor = NSColor.black.withAlphaComponent(0.62).cgColor
        scroll.layer?.borderColor = Palette.violet.withAlphaComponent(0.7).cgColor
        scroll.layer?.borderWidth = 1

        logView = NSTextView()
        logView.isEditable = false
        logView.drawsBackground = false
        logView.textColor = Palette.muted
        logView.insertionPointColor = Palette.cyan
        logView.font = .monospacedSystemFont(ofSize: 11, weight: .regular)
        logView.textContainerInset = NSSize(width: 14, height: 14)
        logView.string = "Starting private runtime…\n"
        scroll.documentView = logView

        let appDataButton = makeButton(
            title: "OPEN APP DATA",
            action: #selector(openRuntimeFolder),
            accent: Palette.violet
        )

        for view in [title, caption, scroll, appDataButton] {
            container.addSubview(view)
        }
        NSLayoutConstraint.activate([
            title.topAnchor.constraint(equalTo: container.topAnchor, constant: 30),
            title.leadingAnchor.constraint(equalTo: container.leadingAnchor, constant: 34),
            title.trailingAnchor.constraint(equalTo: container.trailingAnchor, constant: -34),
            caption.topAnchor.constraint(equalTo: title.bottomAnchor, constant: 8),
            caption.leadingAnchor.constraint(equalTo: title.leadingAnchor),
            caption.trailingAnchor.constraint(equalTo: title.trailingAnchor),
            scroll.topAnchor.constraint(equalTo: caption.bottomAnchor, constant: 20),
            scroll.leadingAnchor.constraint(equalTo: title.leadingAnchor),
            scroll.trailingAnchor.constraint(equalTo: title.trailingAnchor),
            scroll.bottomAnchor.constraint(equalTo: appDataButton.topAnchor, constant: -18),
            appDataButton.leadingAnchor.constraint(equalTo: title.leadingAnchor),
            appDataButton.bottomAnchor.constraint(equalTo: container.bottomAnchor, constant: -26),
        ])
        return container
    }

    private func makeButton(
        title: String,
        action: Selector,
        accent: NSColor
    ) -> NSButton {
        let button = NSButton(title: title, target: self, action: action)
        button.font = .monospacedSystemFont(ofSize: 10, weight: .bold)
        button.isBordered = false
        button.contentTintColor = Palette.text
        button.wantsLayer = true
        button.layer?.backgroundColor = accent.withAlphaComponent(0.22).cgColor
        button.layer?.borderColor = accent.withAlphaComponent(0.92).cgColor
        button.layer?.borderWidth = 1
        button.layer?.cornerRadius = 2
        button.translatesAutoresizingMaskIntoConstraints = false
        button.heightAnchor.constraint(equalToConstant: 34).isActive = true
        button.widthAnchor.constraint(greaterThanOrEqualToConstant: 88).isActive = true
        return button
    }

    private func setContentMode(_ mode: ContentMode) {
        contentMode = mode
        loadingView.isHidden = mode != .loading
        webView.isHidden = mode != .cleaner
        diagnosticsView.isHidden = mode != .diagnostics
        switch mode {
        case .diagnostics:
            diagnosticsButton.title = uiURL == nil ? "BACK TO STARTUP" : "BACK TO CLEANER"
        default:
            diagnosticsButton.title = "DIAGNOSTICS"
        }
    }

    private func appendLog(_ text: String) {
        let logDirectory = launcherLogURL.deletingLastPathComponent()
        try? FileManager.default.createDirectory(
            at: logDirectory,
            withIntermediateDirectories: true
        )
        if !FileManager.default.fileExists(atPath: launcherLogURL.path) {
            FileManager.default.createFile(atPath: launcherLogURL.path, contents: nil)
        }
        if let data = text.data(using: .utf8),
           let handle = try? FileHandle(forWritingTo: launcherLogURL) {
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: data)
            try? handle.close()
        }

        DispatchQueue.main.async {
            self.logView.textStorage?.append(NSAttributedString(string: text))
            self.logView.scrollToEndOfDocument(nil)
            self.pendingBackendOutput += text
            while let newline = self.pendingBackendOutput.firstIndex(of: "\n") {
                let line = String(self.pendingBackendOutput[..<newline])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                self.pendingBackendOutput.removeSubrange(...newline)
                self.handleBackendLine(line)
            }
        }
    }

    private func handleBackendLine(_ line: String) {
        if line.hasPrefix("[1/5]") {
            statusLabel.stringValue = "PREPARING PRIVATE PYTHON RUNTIME…"
        } else if line.hasPrefix("[2/5]") {
            statusLabel.stringValue = "INSTALLING LOCAL VIDEO COMPONENTS…"
        } else if line.hasPrefix("[3/5]") {
            statusLabel.stringValue = "CONFIGURING VIDEO TOOLS…"
        } else if line.hasPrefix("[4/5]") {
            statusLabel.stringValue = "PREPARING BOTH RESTORATION MODELS…"
        } else if line.hasPrefix("[5/5]") {
            statusLabel.stringValue = "STARTING CLEANER…"
        } else if line.hasPrefix("MAC_WATERMARK_MODEL_PROGRESS=") {
            let payload = line.dropFirst("MAC_WATERMARK_MODEL_PROGRESS=".count)
            let fields = payload.split(separator: "|", maxSplits: 1)
            if fields.count == 2, let fraction = Double(fields[0]) {
                let percent = Int((min(1, max(0, fraction)) * 100).rounded())
                statusLabel.stringValue = "\(percent)% // \(fields[1])"
            }
        } else if line.hasPrefix("MAC_WATERMARK_UI_URL="),
                  let url = URL(string: String(line.dropFirst("MAC_WATERMARK_UI_URL=".count))),
                  url.scheme == "http", url.host == "127.0.0.1", url.port != nil {
            uiURL = url
            waitUntilReady(url: url, attempt: 0)
        }
    }

    private func startBackend() {
        let script = resourceURL().appendingPathComponent("bootstrap_macos.sh")
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/bin/bash")
        task.arguments = [script.path, "serve"]
        task.currentDirectoryURL = resourceURL()
        var environment = ProcessInfo.processInfo.environment
        environment["MAC_WATERMARK_SELECTION_FILE"] = selectionManifestURL.path
        environment["MAC_WATERMARK_TRASH_HELPER"] = resourceURL()
            .appendingPathComponent("tools/trash-item").path
        task.environment = environment

        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            self?.appendLog(String(decoding: data, as: UTF8.self))
        }
        task.terminationHandler = { [weak self] process in
            DispatchQueue.main.async {
                guard let self, !self.isTerminating else { return }
                self.spinner.stopAnimation(nil)
                self.statusLabel.stringValue = "CLEANER STOPPED // VIEW DIAGNOSTICS"
                self.appendLog(
                    "\nThe cleaner exited with status \(process.terminationStatus). " +
                    "Review the setup details below, then reopen the cleaner to retry. " +
                    "Your source videos were kept.\n"
                )
                self.setContentMode(.diagnostics)
            }
        }

        do {
            try task.run()
            process = task
        } catch {
            spinner.stopAnimation(nil)
            statusLabel.stringValue = "PRIVATE RUNTIME COULD NOT START"
            appendLog("ERROR: \(error.localizedDescription)\n")
            setContentMode(.diagnostics)
        }
    }

    private func waitUntilReady(url: URL, attempt: Int) {
        guard attempt < 120 else {
            statusLabel.stringValue = "STARTUP TIMED OUT // VIEW DIAGNOSTICS"
            appendLog("ERROR: Local cleaner did not become healthy within 60 seconds.\n")
            setContentMode(.diagnostics)
            return
        }
        let health = url.appendingPathComponent("_stcore/health")
        URLSession.shared.dataTask(with: health) { [weak self] _, response, _ in
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                if let http = response as? HTTPURLResponse, http.statusCode == 200 {
                    guard let self else { return }
                    self.spinner.stopAnimation(nil)
                    self.statusLabel.stringValue = "READY // LOCAL VIDEO RESTORATION"
                    self.loadCleaner(reloadingSelection: false)
                } else {
                    self?.waitUntilReady(url: url, attempt: attempt + 1)
                }
            }
        }.resume()
    }

    private func loadCleaner(reloadingSelection: Bool) {
        guard let uiURL else { return }
        var components = URLComponents(url: uiURL, resolvingAgainstBaseURL: false)
        if reloadingSelection {
            components?.queryItems = [
                URLQueryItem(
                    name: "selection",
                    value: String(Int(Date().timeIntervalSince1970 * 1000))
                ),
            ]
        }
        let url = components?.url ?? uiURL
        setContentMode(.cleaner)
        webView.load(URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData))
    }

    private func refreshSelectionCount() {
        guard let data = try? Data(contentsOf: selectionManifestURL),
              let payload = try? JSONDecoder().decode(
                  SelectionManifest.self,
                  from: data
              ) else {
            selectionLabel.stringValue = "NO SOURCE QUEUE"
            return
        }
        let valid = payload.sources.filter { record in
            (try? SourceRecord.capture(URL(fileURLWithPath: record.path))) == record
        }
        selectionLabel.stringValue =
            valid.count == 1 ? "1 VIDEO QUEUED" : "\(valid.count) VIDEOS QUEUED"
    }

    private func withSelectionManifestLock<T>(
        _ body: () throws -> T
    ) throws -> T {
        let lockURL = selectionManifestURL.appendingPathExtension("lock")
        try FileManager.default.createDirectory(
            at: lockURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let descriptor = lockURL.path.withCString { pointer in
            Darwin.open(pointer, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR)
        }
        guard descriptor >= 0 else {
            throw NSError(
                domain: NSPOSIXErrorDomain,
                code: Int(errno),
                userInfo: [
                    NSLocalizedDescriptionKey: "Could not lock the source queue.",
                ]
            )
        }
        defer {
            _ = flock(descriptor, LOCK_UN)
            _ = Darwin.close(descriptor)
        }
        guard flock(descriptor, LOCK_EX) == 0 else {
            throw NSError(
                domain: NSPOSIXErrorDomain,
                code: Int(errno),
                userInfo: [
                    NSLocalizedDescriptionKey: "Could not lock the source queue.",
                ]
            )
        }
        return try body()
    }

    @objc private func chooseVideos() {
        let panel = NSOpenPanel()
        panel.title = "Choose videos to clean"
        panel.prompt = "Add to Cleaner"
        panel.message =
            "The cleaner saves a new copy and keeps your originals."
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = true
        panel.allowedContentTypes = supportedVideoExtensions.compactMap {
            UTType(filenameExtension: $0)
        }

        panel.beginSheetModal(for: window) { [weak self] response in
            guard let self, response == .OK, !panel.urls.isEmpty else { return }
            do {
                let records = try panel.urls.map(SourceRecord.capture)
                let unique = Array(
                    Dictionary(
                        records.map { ("\($0.device):\($0.inode)", $0) },
                        uniquingKeysWith: { first, _ in first }
                    ).values
                ).sorted { $0.path.localizedStandardCompare($1.path) == .orderedAscending }
                let manifest = SelectionManifest(
                    version: 2,
                    selected_at: ISO8601DateFormatter().string(from: Date()),
                    sources: unique
                )
                try self.withSelectionManifestLock {
                    let data = try JSONEncoder().encode(manifest)
                    try data.write(
                        to: self.selectionManifestURL,
                        options: .atomic
                    )
                }
                self.refreshSelectionCount()
                if self.uiURL != nil {
                    self.loadCleaner(reloadingSelection: true)
                }
            } catch {
                self.appendLog("ERROR: Could not save source queue: \(error)\n")
                self.setContentMode(.diagnostics)
            }
        }
    }

    @objc private func toggleDiagnostics() {
        if contentMode == .diagnostics {
            setContentMode(uiURL == nil ? .loading : .cleaner)
        } else {
            setContentMode(.diagnostics)
        }
    }

    @objc private func openOutputFolder() {
        let path = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Movies/MAC WATERMARK CLEANER", isDirectory: true)
        try? FileManager.default.createDirectory(
            at: path,
            withIntermediateDirectories: true
        )
        NSWorkspace.shared.open(path)
    }

    @objc private func openRuntimeFolder() {
        try? FileManager.default.createDirectory(
            at: runtimeRootURL,
            withIntermediateDirectories: true
        )
        NSWorkspace.shared.open(runtimeRootURL)
    }

    func webView(
        _ webView: WKWebView,
        decidePolicyFor navigationAction: WKNavigationAction,
        decisionHandler: @escaping (WKNavigationActionPolicy) -> Void
    ) {
        guard let destination = navigationAction.request.url,
              let uiURL else {
            decisionHandler(.allow)
            return
        }
        let isLocal =
            destination.host == uiURL.host
            && destination.port == uiURL.port
        if navigationAction.shouldPerformDownload && (isLocal || destination.scheme == "blob") {
            decisionHandler(.download)
            return
        }
        if navigationAction.navigationType == .linkActivated && !isLocal {
            NSWorkspace.shared.open(destination)
            decisionHandler(.cancel)
        } else {
            decisionHandler(.allow)
        }
    }

    func webView(
        _ webView: WKWebView,
        decidePolicyFor navigationResponse: WKNavigationResponse,
        decisionHandler: @escaping (WKNavigationResponsePolicy) -> Void
    ) {
        let response = navigationResponse.response as? HTTPURLResponse
        let attachment = response?.value(forHTTPHeaderField: "Content-Disposition")?
            .lowercased().contains("attachment") == true
        decisionHandler(attachment || !navigationResponse.canShowMIMEType ? .download : .allow)
    }

    func webView(_ webView: WKWebView, navigationAction: WKNavigationAction, didBecome download: WKDownload) {
        download.delegate = self
    }

    func webView(_ webView: WKWebView, navigationResponse: WKNavigationResponse, didBecome download: WKDownload) {
        download.delegate = self
    }

    func download(
        _ download: WKDownload,
        decideDestinationUsing response: URLResponse,
        suggestedFilename: String,
        completionHandler: @escaping (URL?) -> Void
    ) {
        let directory = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Downloads")
        let filename = URL(fileURLWithPath: suggestedFilename).lastPathComponent
        let original = directory.appendingPathComponent(filename.isEmpty ? "cleaned-video.mp4" : filename)
        var destination = original
        var suffix = 2
        while FileManager.default.fileExists(atPath: destination.path) {
            destination = directory.appendingPathComponent(
                "\(original.deletingPathExtension().lastPathComponent) (\(suffix)).\(original.pathExtension)"
            )
            suffix += 1
        }
        downloadDestinations[ObjectIdentifier(download)] = destination
        completionHandler(destination)
    }

    func downloadDidFinish(_ download: WKDownload) {
        guard let destination = downloadDestinations.removeValue(forKey: ObjectIdentifier(download)) else { return }
        appendLog("Saved download: \(destination.path)\n")
        NSWorkspace.shared.activateFileViewerSelecting([destination])
    }

    func download(_ download: WKDownload, didFailWithError error: Error, resumeData: Data?) {
        downloadDestinations.removeValue(forKey: ObjectIdentifier(download))
        appendLog("Download failed: \(error.localizedDescription)\n")
        let alert = NSAlert()
        alert.messageText = "The download could not be saved."
        alert.informativeText = "Your cleaned video is still in the OUTPUT folder. \(error.localizedDescription)"
        alert.beginSheetModal(for: window)
    }
}

let application = NSApplication.shared
let delegate = AppDelegate()
application.delegate = delegate
application.run()
