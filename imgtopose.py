import os
import argparse
import subprocess
from extract_frames import extract_frames
from llff.poses.pose_utils import gen_poses

def run_colmap(scene_dir, match_type='exhaustive_matcher'):
    print(f"[INFO] Running COLMAP with match_type: {match_type}")
    subprocess.run([
        "colmap", "automatic_reconstructor",
        "--workspace_path", scene_dir,
        "--image_path", os.path.join(scene_dir, "images"),
        "--data_type", "individual",
        "--single_camera", "1",
        "--quality", "high",
        "--use_gpu", "0",
    ], check=True)

def main(video_path, scene_dir, num_frames, resize, match_type):
    images_dir = os.path.join(scene_dir, 'images')
    print(f"[INFO] Extracting {num_frames} frames from {video_path} into {images_dir}")
    if resize:
        extract_frames(video_path, images_dir, num_frames, resize_width=resize[0], resize_height=resize[1])
    else:
        extract_frames(video_path, images_dir, num_frames)

    # (Optional) Run SAM alpha mask generation
    masks_dir = os.path.join(scene_dir, 'masks')
    run_sam_masks(images_dir, masks_dir)

    # Run COLMAP + poses extraction
    print("[INFO] Generating poses using COLMAP and LLFF utils")
    gen_poses(scene_dir, match_type, factors=[4])  # factor 4 downsample like LLFF default

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Input video path")
    parser.add_argument("scene_dir", help="Scene directory to save outputs")
    parser.add_argument("--num_frames", type=int, default=50)
    parser.add_argument("--resize", nargs=2, type=int, metavar=('width', 'height'), help="Resize frames")
    parser.add_argument("--match_type", type=str, default='exhaustive_matcher', choices=['exhaustive_matcher', 'sequential_matcher'])

    args = parser.parse_args()

    main(args.video_path, args.scene_dir, args.num_frames, args.resize, args.match_type)
