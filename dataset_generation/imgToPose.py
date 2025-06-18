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

    camera_model = args.get("camera_model", "SIMPLE_RADIAL")
    use_gpu = args.get("use_gpu", False)
    match_type = args.get("match_type", "sequential_matcher")
    vocab_tree = args.get("vocab_tree", None)
    loop_detection = args.get("loop_detection", False)
    mask_folder = args.get("mask_folder", False)

    colmap_dir = os.path.join(basedir, 'colmap')
    os.makedirs(colmap_dir, exist_ok=True)

    db_path = os.path.join(colmap_dir, 'database.db')
    img_path = os.path.join(basedir, 'images')

    if mask_folder:
        mask_folder = os.path.join(basedir, 'masks')
    sparse_path = os.path.join(basedir, 'sparse')
    log_path = os.path.join(colmap_dir, 'colmap_output.txt')

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

        if match_type == 'vocab_tree_matcher':
            match_args += ['--VocabTreeMatching.vocab_tree_path', vocab_tree]

        if not use_gpu:
            match_args += ['--SiftMatching.use_gpu', '0']
        
        # Loop detection (extra step if requested)
        if loop_detection:
            if vocab_tree is None:
                raise ValueError("vocab_tree path must be provided for loop detection")

            loop_args = [
                'colmap', 'vocab_tree_matcher',
                '--database_path', db_path,
                '--VocabTreeMatching.vocab_tree_path', vocab_tree
            ]
            if not use_gpu:
                loop_args += ['--SiftMatching.use_gpu', '0']

            print("Running loop detection")
            subprocess.run(loop_args, stdout=logfile, stderr=logfile)

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

        # Convert binary COLMAP model to TXT format
        convert_colmap_bin_to_txt(sparse_path, logfile)

    print(f"COLMAP logs saved to {log_path}")

def convert_colmap_bin_to_txt(sparse_path, logfile):
    """
    Convert COLMAP sparse model from BIN to TXT format. .txt files are used in old version of colmap which colmap2nerf.py expects.
    
    Parameters:
    - sparse_path: path to COLMAP sparse folder containing '0' subfolder with .bin files
    - logfile: open file object for logging subprocess output
    """
    bin_model_path = os.path.join(sparse_path, '0')
    print("COLMAP model from BIN to TXT")
    convert_args = [
        'colmap', 'model_converter',
        '--input_path', bin_model_path,
        '--output_path', bin_model_path,
        '--output_type', 'TXT'
    ]
    subprocess.run(convert_args, stdout=logfile, stderr=logfile)

def convert_colmap_to_nerf(basedir):
    # Convert COLMAP sparse reconstruction to NeRF format
    colmap_sparse = os.path.join(basedir, 'sparse', '0')
    image_folder = os.path.join(basedir, 'images')
    output_file = os.path.join(basedir, 'transforms.json')

    cmd = [
        "python", "/Users/mackenzie/Documents/GitHub/mesh_reconstruction/tools/colmap2nerf.py",
        "--text", colmap_sparse,
        "--images", image_folder,
        "--out", output_file
    ]

    subprocess.run(cmd)

def main():
    # Base dataset directory
    basedir = "/Users/mackenzie/Documents/GitHub/mesh_reconstruction/data/grogu"  

    # Define COLMAP configuration
    args = {
        "camera_model": "SIMPLE_RADIAL_FISHEYE", 
        "use_gpu": True,
        "match_type": "sequential_matcher", 
        "vocab_tree": './vocab_tree_flickr100K_words256K.bin', # if using vocab matching
        "loop_detection": True,
        "masks" : True,  # Set to True if you have masks for images 
    }

    print("Running COLMAP SFM")
    run_colmap(basedir, args)

    print("Convert COLMAP to NERF")
    convert_colmap_to_nerf(basedir)

    print("Done")


if __name__ == "__main__":
    main()
