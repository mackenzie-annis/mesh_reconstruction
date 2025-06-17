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

    camera_model = args.get("camera_model", "SIMPLE_RADIAL")
    use_gpu = args.get("use_gpu", False)
    match_type = args.get("match_type", "sequential_matcher")
    vocab_tree = args.get("vocab_tree", None)
    loop_detection = args.get("loop_detection", False)
    mask_folder = args.get("mask_folder", None)

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
        if not use_gpu:
            feature_args += ['--SiftExtraction.use_gpu', '0']

        print("Running feature_extractor")
        subprocess.run(feature_args, stdout=logfile, stderr=logfile)

        # Matching
        match_args = [
            'colmap', match_type,
            '--database_path', db_path
        ]
        if vocab_tree:
            match_args += ['--VocabTreeMatching.vocab_tree_path', vocab_tree]
        if not use_gpu:
            match_args += ['--SiftMatching.use_gpu', '0']

        print("Running matcher")
        subprocess.run(match_args, stdout=logfile, stderr=logfile)

        # Mapping
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
        subprocess.run(mapper_args, stdout=logfile, stderr=logfile)

    print(f"COLMAP logs saved to {log_path}")

def convert_colmap_to_nerf(basedir):
    # Convert COLMAP sparse reconstruction to NeRF format
    colmap_sparse = os.path.join(basedir, 'sparse', '0')
    image_folder = os.path.join(basedir, 'images')
    output_dir = os.path.join(basedir, 'nerf_data')

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "python", "/Users/mackenzie/Documents/GitHub/mesh_reconstruction/tools/colmap2nerf.py",
        "--colmap_dir", colmap_sparse,
        "--image_folder", image_folder,
        "--output_dir", output_dir
    ]

    subprocess.run(cmd)


