import os
import subprocess

def run_colmap(basedir, args):
    """
    Run COLMAP pipeline using settings from args dictionary.

    Parameters:
    - basedir: root directory where database, images, sparse output, etc. are stored
    - args: dictionary with keys:
        - camera_model: ie 'SIMPLE_RADIAL', 'PINHOLE', etc
        - use_gpu (bool)
        - match_type: 'exhaustive_matcher', 'sequential_matcher', or 'vocab_tree_matcher'
        - vocab_tree: reqired for loop_detection or vocab_tree_matcher, path to vocab tree file
        - image_type: video, image
        - loop_detection: bool, whether to run loop detection
    """

    # Load parameters from args
    camera_model = args.get("camera_model", "SIMPLE_RADIAL")
    use_gpu = args.get("use_gpu", False)
    match_type = args.get("match_type", "sequential_matcher")
    vocab_tree = args.get("vocab_tree", None)
    loop_detection = args.get("loop_detection", False)

    db_path = os.path.join(basedir, 'database.db')
    img_path = os.path.join(basedir, 'images')
    sparse_path = os.path.join(basedir, 'sparse')
    log_path = os.path.join(basedir, 'colmap_output.txt')

    os.makedirs(sparse_path, exist_ok=True)

    with open(log_path, 'w') as logfile:
        # Feature extraction
        feature_args = [
            'colmap', 'feature_extractor',
            '--database_path', db_path,
            '--image_path', img_path,
            '--ImageReader.camera_model', camera_model,
            '--ImageReader.single_camera', '1',
        ]
        if mask_folder:
            feature_args += ['--ImageReader.mask_path', mask_folder]

        print("Running feature_extractor")
        logfile.write(subprocess.check_output(feature_args, universal_newlines=True))

        # Feature matching
        match_args = [
            'colmap', match_type,
            '--database_path', db_path
        ]
        if vocab_tree:
            match_args += ['--VocabTreeMatching.vocab_tree_path', vocab_tree]

        print("Running matcher")
        logfile.write(subprocess.check_output(match_args, universal_newlines=True))

        # Sparse mapping
        mapper_args = [
            'colmap', 'mapper',
            '--database_path', db_path,
            '--image_path', img_path,
            '--output_path', sparse_path,
            '--Mapper.num_threads', '8',
            '--Mapper.init_min_tri_angle', '4',
            '--Mapper.multiple_models', '0',
            '--Mapper.extract_colors', '0'
        ]

        print("Running sparse mapper")
        logfile.write(subprocess.check_output(mapper_args, universal_newlines=True))
        logfile.close()

    print(f"Logs saved to {log_path}")



