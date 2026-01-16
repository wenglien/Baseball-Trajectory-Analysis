import SwiftUI
import AVFoundation

struct ContentView: View {
    @StateObject private var viewModel = CameraViewModel()
    
    var body: some View {
        ZStack {
            CameraPreview(session: viewModel.captureSession)
                .ignoresSafeArea()
            
            PoseAndBallOverlay(poseLandmarks: viewModel.poseLandmarks,
                               ballBoxes: viewModel.ballBoxes,
                               frameSize: viewModel.frameSize)
            
            VStack {
                HStack {
                    Spacer()
                    VStack(alignment: .trailing) {
                        if let speed = viewModel.currentSpeedKmh {
                            Text(String(format: "%.1f km/h", speed))
                                .font(.system(size: 28, weight: .bold))
                                .padding(8)
                                .background(Color.black.opacity(0.6))
                                .foregroundColor(.white)
                                .cornerRadius(8)
                        } else {
                            Text("--.- km/h")
                                .font(.system(size: 28, weight: .bold))
                                .padding(8)
                                .background(Color.black.opacity(0.4))
                                .foregroundColor(.white)
                                .cornerRadius(8)
                        }
                        
                        if let releaseInfo = viewModel.lastPitchInfo {
                            Text("Δt: \(String(format: "%.3f s", releaseInfo.deltaTime))")
                                .font(.caption)
                                .padding(4)
                                .background(Color.black.opacity(0.4))
                                .foregroundColor(.white)
                                .cornerRadius(4)
                        }
                    }
                    .padding()
                }
                Spacer()
            }
        }
        .onAppear {
            viewModel.startSession()
        }
        .onDisappear {
            viewModel.stopSession()
        }
    }
}

