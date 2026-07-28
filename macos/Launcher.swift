import AppKit
import Darwin
import Foundation

private let appName = "MAC WATERMARK CLEANER"

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
        resources.appendingPathComponent("engine/requirements-macos.lock"),
        resources.appendingPathComponent("THIRD_PARTY_NOTICES.md"),
    ]
    for path in required where !FileManager.default.fileExists(atPath: path.path) {
        fputs("SELF_TEST_FAILED missing \(path.path)\n", stderr)
        exit(1)
    }
    print("SELF_TEST_OK \(resources.path)")
    exit(0)
}

if CommandLine.arguments.contains("--self-test") {
    runSelfTest()
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var window: NSWindow!
    private var statusLabel: NSTextField!
    private var logView: NSTextView!
    private var spinner: NSProgressIndicator!
    private var openButton: NSButton!
    private var process: Process?
    private var uiURL: URL?
    private lazy var launcherLogURL: URL = {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(
                "Library/Application Support/MAC WATERMARK CLEANER/Logs/launcher.log"
            )
    }()

    func applicationDidFinishLaunching(_ notification: Notification) {
        buildWindow()
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        window.makeKeyAndOrderFront(nil)
        startBackend()
    }

    func applicationWillTerminate(_ notification: Notification) {
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
            contentRect: NSRect(x: 0, y: 0, width: 720, height: 480),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = appName
        window.center()
        window.minSize = NSSize(width: 620, height: 400)

        let content = NSView()
        content.translatesAutoresizingMaskIntoConstraints = false
        window.contentView = content

        let title = NSTextField(labelWithString: appName)
        title.font = .systemFont(ofSize: 25, weight: .bold)
        title.translatesAutoresizingMaskIntoConstraints = false

        let subtitle = NSTextField(
            wrappingLabelWithString:
                "Universal DeMark-World detection with a private, Mac-safe runtime."
        )
        subtitle.textColor = .secondaryLabelColor
        subtitle.translatesAutoresizingMaskIntoConstraints = false

        spinner = NSProgressIndicator()
        spinner.style = .spinning
        spinner.controlSize = .small
        spinner.startAnimation(nil)
        spinner.translatesAutoresizingMaskIntoConstraints = false

        statusLabel = NSTextField(labelWithString: "Preparing first launch…")
        statusLabel.font = .systemFont(ofSize: 13, weight: .medium)
        statusLabel.translatesAutoresizingMaskIntoConstraints = false

        let scroll = NSScrollView()
        scroll.hasVerticalScroller = true
        scroll.borderType = .bezelBorder
        scroll.translatesAutoresizingMaskIntoConstraints = false
        logView = NSTextView()
        logView.isEditable = false
        logView.font = .monospacedSystemFont(ofSize: 11, weight: .regular)
        logView.string = "Starting…\n"
        scroll.documentView = logView

        openButton = NSButton(
            title: "Open Cleaner",
            target: self,
            action: #selector(openInterface)
        )
        openButton.bezelStyle = .rounded
        openButton.isEnabled = false
        openButton.translatesAutoresizingMaskIntoConstraints = false

        let outputButton = NSButton(
            title: "Open Output Folder",
            target: self,
            action: #selector(openOutputFolder)
        )
        outputButton.translatesAutoresizingMaskIntoConstraints = false

        let runtimeButton = NSButton(
            title: "Open App Data",
            target: self,
            action: #selector(openRuntimeFolder)
        )
        runtimeButton.translatesAutoresizingMaskIntoConstraints = false

        let views: [NSView?] = [
            title, subtitle, spinner, statusLabel, scroll,
            openButton, outputButton, runtimeButton,
        ]
        for view in views.compactMap({ $0 }) {
            content.addSubview(view)
        }

        NSLayoutConstraint.activate([
            title.topAnchor.constraint(equalTo: content.topAnchor, constant: 24),
            title.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 26),
            title.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -26),

            subtitle.topAnchor.constraint(equalTo: title.bottomAnchor, constant: 5),
            subtitle.leadingAnchor.constraint(equalTo: title.leadingAnchor),
            subtitle.trailingAnchor.constraint(equalTo: title.trailingAnchor),

            spinner.topAnchor.constraint(equalTo: subtitle.bottomAnchor, constant: 18),
            spinner.leadingAnchor.constraint(equalTo: title.leadingAnchor),
            statusLabel.centerYAnchor.constraint(equalTo: spinner.centerYAnchor),
            statusLabel.leadingAnchor.constraint(equalTo: spinner.trailingAnchor, constant: 9),
            statusLabel.trailingAnchor.constraint(equalTo: title.trailingAnchor),

            scroll.topAnchor.constraint(equalTo: spinner.bottomAnchor, constant: 14),
            scroll.leadingAnchor.constraint(equalTo: title.leadingAnchor),
            scroll.trailingAnchor.constraint(equalTo: title.trailingAnchor),
            scroll.bottomAnchor.constraint(equalTo: openButton.topAnchor, constant: -18),

            openButton.leadingAnchor.constraint(equalTo: title.leadingAnchor),
            openButton.bottomAnchor.constraint(equalTo: content.bottomAnchor, constant: -22),
            outputButton.leadingAnchor.constraint(equalTo: openButton.trailingAnchor, constant: 10),
            outputButton.centerYAnchor.constraint(equalTo: openButton.centerYAnchor),
            runtimeButton.leadingAnchor.constraint(equalTo: outputButton.trailingAnchor, constant: 10),
            runtimeButton.centerYAnchor.constraint(equalTo: openButton.centerYAnchor),
        ])
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
            if text.contains("[1/4]") {
                self.statusLabel.stringValue = "Preparing private Python runtime…"
            } else if text.contains("[2/4]") {
                self.statusLabel.stringValue = "Installing app components…"
            } else if text.contains("[3/4]") {
                self.statusLabel.stringValue = "Configuring video tools…"
            } else if text.contains("[4/4]") {
                self.statusLabel.stringValue = "Starting the cleaner…"
            }
            for line in text.split(separator: "\n") {
                if line.hasPrefix("MAC_WATERMARK_UI_URL="),
                   let url = URL(
                       string: String(line.dropFirst("MAC_WATERMARK_UI_URL=".count))
                   ) {
                    self.uiURL = url
                    self.waitUntilReady(url: url, attempt: 0)
                }
            }
        }
    }

    private func startBackend() {
        let script = resourceURL().appendingPathComponent("bootstrap_macos.sh")
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/bin/bash")
        task.arguments = [script.path, "serve"]
        task.currentDirectoryURL = resourceURL()

        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else {
                return
            }
            self?.appendLog(text)
        }
        task.terminationHandler = { [weak self] process in
            DispatchQueue.main.async {
                guard let self else { return }
                self.spinner.stopAnimation(nil)
                if process.terminationStatus != 0 {
                    self.statusLabel.stringValue =
                        "Setup stopped. Read the last error below, then reopen the app."
                    self.appendLog(
                        "\nThe cleaner exited with status \(process.terminationStatus). " +
                        "First-time setup requires internet access; originals were untouched.\n"
                    )
                }
            }
        }

        do {
            try task.run()
            process = task
        } catch {
            spinner.stopAnimation(nil)
            statusLabel.stringValue = "Could not start the private runtime."
            appendLog("ERROR: \(error.localizedDescription)\n")
        }
    }

    private func waitUntilReady(url: URL, attempt: Int) {
        guard attempt < 60 else {
            statusLabel.stringValue = "Server is taking longer than expected. Check the log."
            return
        }
        URLSession.shared.dataTask(with: url) { [weak self] _, response, _ in
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                if let http = response as? HTTPURLResponse, (200..<500).contains(http.statusCode) {
                    guard let self else { return }
                    self.spinner.stopAnimation(nil)
                    self.statusLabel.stringValue = "Ready. Originals are never deleted."
                    self.openButton.isEnabled = true
                    self.openInterface()
                } else {
                    self?.waitUntilReady(url: url, attempt: attempt + 1)
                }
            }
        }.resume()
    }

    @objc private func openInterface() {
        if let uiURL {
            NSWorkspace.shared.open(uiURL)
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
        let path = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(
                "Library/Application Support/MAC WATERMARK CLEANER",
                isDirectory: true
            )
        try? FileManager.default.createDirectory(
            at: path,
            withIntermediateDirectories: true
        )
        NSWorkspace.shared.open(path)
    }
}

let application = NSApplication.shared
let delegate = AppDelegate()
application.delegate = delegate
application.run()
