import SwiftUI

@main
struct NegPyApp: App {
    @StateObject private var library = Library()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(library)
                .preferredColorScheme(.dark)
        }
    }
}

struct RootView: View {
    @EnvironmentObject var library: Library

    var body: some View {
        NavigationStack {
            if library.folder == nil {
                WelcomeView()
            } else {
                RollListView()
            }
        }
        .task { library.restore() }
    }
}

struct WelcomeView: View {
    @EnvironmentObject var library: Library
    @State private var picking = false

    var body: some View {
        VStack(spacing: 24) {
            Image(systemName: "film")
                .font(.system(size: 64))
            Text("NegPy")
                .font(.largeTitle.bold())
            Text("Open the NegPy folder in iCloud Drive. The Mac writes every edit and preview there; this device reads them and writes its own edits back.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
                .padding(.horizontal)
            Button("Open Sync Folder…") { picking = true }
                .buttonStyle(.borderedProminent)
        }
        .padding()
        .sheet(isPresented: $picking) {
            FolderPicker { url in library.adopt(folder: url) }
        }
    }
}
