import SwiftUI

struct PoseAndBallOverlay: View {
    let poseLandmarks: [PoseLandmark]
    let ballBoxes: [CGRect]
    let frameSize: CGSize
    
    private let linePairs: [(Int, Int)] = [
        // 這裡放一些常見的骨架連線（以 MediaPipe 33 點 index 為例）
        (11, 13), (13, 15), // 左手
        (12, 14), (14, 16), // 右手
        (11, 12),           // 肩膀
        (23, 24),           // 臀部
        (11, 23), (12, 24), // 上身連接
        (23, 25), (25, 27), // 左腳
        (24, 26), (26, 28)  // 右腳
    ]
    
    var body: some View {
        GeometryReader { geo in
            let size = geo.size
            ZStack {
                drawPose(in: size)
                drawBalls(in: size)
            }
        }
        .allowsHitTesting(false)
    }
    
    private func drawPose(in size: CGSize) -> some View {
        let points = poseLandmarks.map { lm -> CGPoint in
            let x = lm.x * size.width
            let y = lm.y * size.height
            return CGPoint(x: x, y: y)
        }
        
        return ZStack {
            // 骨架線
            Path { path in
                for (startIdx, endIdx) in linePairs {
                    guard startIdx < points.count, endIdx < points.count else { continue }
                    let p1 = points[startIdx]
                    let p2 = points[endIdx]
                    path.move(to: p1)
                    path.addLine(to: p2)
                }
            }
            .stroke(Color.green.opacity(0.8), lineWidth: 3)
            
            // 關節點
            ForEach(Array(points.enumerated()), id: \.offset) { _, point in
                Circle()
                    .fill(Color.yellow)
                    .frame(width: 6, height: 6)
                    .position(point)
            }
        }
    }
    
    private func drawBalls(in size: CGSize) -> some View {
        ZStack {
            ForEach(0..<ballBoxes.count, id: \.self) { idx in
                let box = ballBoxes[idx]
                let rect = CGRect(x: box.origin.x * size.width,
                                  y: box.origin.y * size.height,
                                  width: box.size.width * size.width,
                                  height: box.size.height * size.height)
                RoundedRectangle(cornerRadius: 4)
                    .stroke(Color.red, lineWidth: 3)
                    .frame(width: rect.width, height: rect.height)
                    .position(x: rect.midX, y: rect.midY)
            }
        }
    }
}

