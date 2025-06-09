import cv2
import os
import argparse
import numpy as np

def extract_frames(video_path, output_dir, num_frames, resize_width=None, resize_height=None):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if num_frames > total_frames:
        print(f"Video only has {total_frames} frames. Reducing num_frames.")
        num_frames = total_frames

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame at index {frame_idx}")
            continue

        if resize_width and resize_height:
            frame = cv2.resize(frame, (resize_width, resize_height))

        filename = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(filename, frame)

    cap.release()
    print(f"Saved {num_frames} frames to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract N evenly spaced frames from a video.")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("output_dir", help="Directory to save extracted frames")
    parser.add_argument("--num", type=int, default=50, help="Number of frames to extract (default: 50)")
    parser.add_argument("--resize", nargs=2, type=int, metavar=('width', 'height'),
                        help="Resize frames to given width and height (e.g., --resize 1280 720)")

    args = parser.parse_args()

    if args.resize:
        extract_frames(args.video_path, args.output_dir, args.num, args.resize[0], args.resize[1])
    else:
        extract_frames(args.video_path, args.output_dir, args.num)


