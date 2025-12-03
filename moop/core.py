#!/usr/bin/env python3

import os
import sys
import shutil
import onnxruntime

# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith("--execution-provider") for arg in sys.argv):
    os.environ["OMP_NUM_THREADS"] = "1"
import warnings
from typing import List
import platform
import signal
import pathlib
import argparse

from time import time

import moop.globals
import moop.metadata
import moop.utilities as util
import moop.util_ffmpeg as ffmpeg
import ui.main as main
from settings import Settings
from moop.face_util import extract_face_images
from moop.ProcessEntry import ProcessEntry
from moop.ProcessMgr import ProcessMgr
from moop.ProcessOptions import ProcessOptions
from moop.capturer import get_video_frame_total, release_video

use_torch = False

# cek apakah perlu torch
if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
    use_torch = True

if use_torch:
    import torch
else:
    torch = None


clip_text = None

call_display_ui = None

process_mgr = None


if "ROCMExecutionProvider" in moop.globals.execution_providers:
    del torch

warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    moop.globals.headless = False

    program = argparse.ArgumentParser(
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100)
    )
    program.add_argument(
        "--server_share",
        help="Public server",
        dest="server_share",
        action="store_true",
        default=False,
    )
    program.add_argument(
        "--cuda_device_id",
        help="Index of the cuda gpu to use",
        dest="cuda_device_id",
        type=int,
        default=0,
    )
    moop.globals.startup_args = program.parse_args()
    # Always enable all processors when using GUI
    moop.globals.frame_processors = ["face_swapper", "face_enhancer"]


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [
        execution_provider.replace("ExecutionProvider", "").lower()
        for execution_provider in execution_providers
    ]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    list_providers = [
        provider
        for provider, encoded_execution_provider in zip(
            onnxruntime.get_available_providers(),
            encode_execution_providers(onnxruntime.get_available_providers()),
        )
        if any(
            execution_provider in encoded_execution_provider
            for execution_provider in execution_providers
        )
    ]

    if torch is not None:
        try:
            for i in range(len(list_providers)):
                if list_providers[i] == "CUDAExecutionProvider":
                    list_providers[i] = (
                        "CUDAExecutionProvider",
                        {"device_id": moop.globals.cuda_device_id},
                    )
                    torch.cuda.set_device(moop.globals.cuda_device_id)
                    break
        except:
            pass

    return list_providers


def suggest_max_memory() -> int:
    if platform.system().lower() == "darwin":
        return 4
    return 16


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if "DmlExecutionProvider" in moop.globals.execution_providers:
        return 1
    if "ROCMExecutionProvider" in moop.globals.execution_providers:
        return 1
    return 8


def limit_resources() -> None:
    # limit memory usage
    if moop.globals.max_memory:
        memory = moop.globals.max_memory * 1024**3
        if platform.system().lower() == "darwin":
            memory = moop.globals.max_memory * 1024**6
        if platform.system().lower() == "windows":
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetProcessWorkingSetSize(
                -1, ctypes.c_size_t(memory), ctypes.c_size_t(memory)
            )
        else:
            import resource

            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def release_resources() -> None:
    import gc

    global process_mgr

    if process_mgr is not None:
        process_mgr.release_resources()
        process_mgr = None

    gc.collect()
    if (
        torch is not None
        and "CUDAExecutionProvider" in moop.globals.execution_providers
        and torch.cuda.is_available()
    ):
        with torch.cuda.device("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status(
            "Python version is not supported - please upgrade to 3.9 or higher."
        )
        return False

    download_directory_path = util.resolve_relative_path("../models")
    util.conditional_download(
        download_directory_path,
        [
            [
                "https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/InSwapper/inswapper_128.onnx",
            ],
            [
                "https://huggingface.co/countfloyd/deepfake/resolve/main/reswapper_128.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/ReSwapper/reswapper_128.onnx",
            ],
            [
                "https://huggingface.co/countfloyd/deepfake/resolve/main/reswapper_256.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/ReSwapper/reswapper_256.onnx",
            ],
            [
                "https://huggingface.co/countfloyd/deepfake/resolve/main/GFPGANv1.4.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/GFPGAN/GFPGANv1.4.onnx",
            ],
            [
                "https://github.com/csxmli2016/DMDNet/releases/download/v1/DMDNet.pth",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/DMDNet/DMDNet.pth",
            ],
            [
                "https://huggingface.co/countfloyd/deepfake/resolve/main/GPEN-BFR-512.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/GPEN/GPEN-BFR-512.onnx",
            ],
            [
                "https://huggingface.co/countfloyd/deepfake/resolve/main/restoreformer_plus_plus.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/RestoreFormer/restoreformer_plus_plus.onnx",
            ],
            [
                "https://huggingface.co/countfloyd/deepfake/resolve/main/xseg.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/xseg.onnx",
            ],
        ],
    )
    download_directory_path = util.resolve_relative_path("../models/CLIP")
    util.conditional_download(
        download_directory_path,
        [
            [
                "https://huggingface.co/countfloyd/deepfake/resolve/main/rd64-uni-refined.pth",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/rd64-uni-refined.pth",
            ]
        ],
    )
    download_directory_path = util.resolve_relative_path("../models/buffalo_l")
    util.conditional_download(
        download_directory_path,
        [
            [
                "https://huggingface.co/halllooo/buffalo_l/resolve/main/1k3d68.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/buffalo_l/1k3d68.onnx",
            ],
            [
                "https://huggingface.co/halllooo/buffalo_l/resolve/main/2d106det.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/buffalo_l/2d106det.onnx",
            ],
            [
                "https://huggingface.co/halllooo/buffalo_l/resolve/main/det_10g.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/buffalo_l/det_10g.onnx",
            ],
            [
                "https://huggingface.co/halllooo/buffalo_l/resolve/main/genderage.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/buffalo_l/genderage.onnx",
            ],
            [
                "https://huggingface.co/halllooo/buffalo_l/resolve/main/w600k_r50.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/buffalo_l/w600k_r50.onnx",
            ],
        ],
    )
    download_directory_path = util.resolve_relative_path("../models/CodeFormer")
    util.conditional_download(
        download_directory_path,
        [
            [
                "https://huggingface.co/countfloyd/deepfake/resolve/main/CodeFormerv0.1.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/CodeFormer/CodeFormerv0.1.onnx",
            ]
        ],
    )
    download_directory_path = util.resolve_relative_path("../models/Frame")
    util.conditional_download(
        download_directory_path,
        [
            [
                "https://huggingface.co/countfloyd/deepfake/resolve/main/deoldify_artistic.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/DeOldify/deoldify_artistic.onnx",
            ],
            [
                "https://huggingface.co/countfloyd/deepfake/resolve/main/deoldify_stable.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/DeOldify/deoldify_stable.onnx",
            ],
            [
                "https://huggingface.co/countfloyd/deepfake/resolve/main/isnet-general-use.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/isnet-general-use.onnx",
            ],
            [
                "https://huggingface.co/countfloyd/deepfake/resolve/main/real_esrgan_x4.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/real_esrgan_x4.onnx",
            ],
            [
                "https://huggingface.co/countfloyd/deepfake/resolve/main/real_esrgan_x2.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/real_esrgan_x2.onnx",
            ],
            [
                "https://huggingface.co/countfloyd/deepfake/resolve/main/lsdir_x4.onnx",
                "https://codeberg.org/roop-unleashed/models/media/branch/main/lsdir_x4.onnx",
            ],
        ],
    )

    if not shutil.which("ffmpeg"):
        update_status("ffmpeg is not installed.")
    return True


def set_display_ui(function):
    global call_display_ui

    call_display_ui = function


def update_status(message: str) -> None:
    global call_display_ui

    print(message)
    if call_display_ui is not None:
        call_display_ui(message)


def start() -> None:
    if moop.globals.headless:
        print("Headless mode currently unsupported - starting UI!")
        # faces = extract_face_images(roop.globals.source_path,  (False, 0))
        # roop.globals.INPUT_FACES.append(faces[roop.globals.source_face_index])
        # faces = extract_face_images(roop.globals.target_path,  (False, util.has_image_extension(roop.globals.target_path)))
        # roop.globals.TARGET_FACES.append(faces[roop.globals.target_face_index])
        # if 'face_enhancer' in roop.globals.frame_processors:
        #     roop.globals.selected_enhancer = 'GFPGAN'

    batch_process_regular(None, False, None)


def get_processing_plugins(masking_engine):
    processors = {"faceswap": {}}
    if masking_engine is not None:
        processors.update({masking_engine: {}})

    if moop.globals.selected_enhancer == "GFPGAN":
        processors.update({"gfpgan": {}})
    elif moop.globals.selected_enhancer == "Codeformer":
        processors.update({"codeformer": {}})
    elif moop.globals.selected_enhancer == "DMDNet":
        processors.update({"dmdnet": {}})
    elif moop.globals.selected_enhancer == "GPEN":
        processors.update({"gpen": {}})
    elif moop.globals.selected_enhancer == "Restoreformer++":
        processors.update({"restoreformer++": {}})
    return processors


def live_swap(frame, options):
    global process_mgr

    if frame is None:
        return frame

    if process_mgr is None:
        process_mgr = ProcessMgr(None)

    #    if len(roop.globals.INPUT_FACESETS) <= selected_index:
    #        selected_index = 0
    process_mgr.initialize(
        moop.globals.INPUT_FACESETS, moop.globals.TARGET_FACES, options
    )
    newframe = process_mgr.process_frame(frame)
    if newframe is None:
        return frame
    return newframe


def batch_process_regular(
    swap_model,
    output_method,
    files: list[ProcessEntry],
    masking_engine: str,
    new_clip_text: str,
    use_new_method,
    imagemask,
    restore_original_mouth,
    num_swap_steps,
    progress,
    selected_index=0,
) -> None:
    global clip_text, process_mgr

    release_resources()
    limit_resources()
    if process_mgr is None:
        process_mgr = ProcessMgr(progress)
    mask = imagemask["layers"][0] if imagemask is not None else None
    if len(moop.globals.INPUT_FACESETS) <= selected_index:
        selected_index = 0
    options = ProcessOptions(
        swap_model,
        get_processing_plugins(masking_engine),
        moop.globals.distance_threshold,
        moop.globals.blend_ratio,
        moop.globals.face_swap_mode,
        selected_index,
        new_clip_text,
        mask,
        num_swap_steps,
        moop.globals.subsample_size,
        False,
        restore_original_mouth,
    )
    process_mgr.initialize(
        moop.globals.INPUT_FACESETS, moop.globals.TARGET_FACES, options
    )
    batch_process(output_method, files, use_new_method)
    return


def batch_process_with_options(files: list[ProcessEntry], options, progress):
    global clip_text, process_mgr

    release_resources()
    limit_resources()
    if process_mgr is None:
        process_mgr = ProcessMgr(progress)
    process_mgr.initialize(
        moop.globals.INPUT_FACESETS, moop.globals.TARGET_FACES, options
    )
    moop.globals.keep_frames = False
    moop.globals.wait_after_extraction = False
    moop.globals.skip_audio = False
    batch_process("Files", files, True)


def batch_process(output_method, files: list[ProcessEntry], use_new_method) -> None:
    global clip_text, process_mgr

    moop.globals.processing = True

    # limit threads for some providers
    max_threads = suggest_execution_threads()
    if max_threads == 1:
        moop.globals.execution_threads = 1

    imagefiles: list[ProcessEntry] = []
    videofiles: list[ProcessEntry] = []

    update_status("Sorting videos/images")

    for index, f in enumerate(files):
        fullname = f.filename
        if util.has_image_extension(fullname):
            destination = util.get_destfilename_from_path(
                fullname,
                moop.globals.output_path,
                f".{moop.globals.CFG.output_image_format}",
            )
            destination = util.replace_template(destination, index=index)
            pathlib.Path(os.path.dirname(destination)).mkdir(
                parents=True, exist_ok=True
            )
            f.finalname = destination
            imagefiles.append(f)

        elif util.is_video(fullname) or util.has_extension(fullname, ["gif"]):
            destination = util.get_destfilename_from_path(
                fullname,
                moop.globals.output_path,
                f"__temp.{moop.globals.CFG.output_video_format}",
            )
            f.finalname = destination
            videofiles.append(f)

    if len(imagefiles) > 0:
        update_status("Processing image(s)")
        origimages = []
        fakeimages = []
        for f in imagefiles:
            origimages.append(f.filename)
            fakeimages.append(f.finalname)

        process_mgr.run_batch(origimages, fakeimages, moop.globals.execution_threads)
        origimages.clear()
        fakeimages.clear()

    if len(videofiles) > 0:
        for index, v in enumerate(videofiles):
            if not moop.globals.processing:
                end_processing("Processing stopped!")
                return
            fps = v.fps if v.fps > 0 else util.detect_fps(v.filename)
            if v.endframe == 0:
                v.endframe = get_video_frame_total(v.filename)

            is_streaming_only = output_method == "Virtual Camera"
            if is_streaming_only == False:
                update_status(
                    f"Creating {os.path.basename(v.finalname)} with {fps} FPS..."
                )

            start_processing = time()
            if (
                is_streaming_only == False
                and moop.globals.keep_frames
                or not use_new_method
            ):
                util.create_temp(v.filename)
                update_status("Extracting frames...")
                ffmpeg.extract_frames(v.filename, v.startframe, v.endframe, fps)
                if not moop.globals.processing:
                    end_processing("Processing stopped!")
                    return

                temp_frame_paths = util.get_temp_frame_paths(v.filename)
                process_mgr.run_batch(
                    temp_frame_paths, temp_frame_paths, moop.globals.execution_threads
                )
                if not moop.globals.processing:
                    end_processing("Processing stopped!")
                    return
                if moop.globals.wait_after_extraction:
                    extract_path = os.path.dirname(temp_frame_paths[0])
                    util.open_folder(extract_path)
                    input("Press any key to continue...")
                    print("Resorting frames to create video")
                    util.sort_rename_frames(extract_path)

                ffmpeg.create_video(v.filename, v.finalname, fps)
                if not moop.globals.keep_frames:
                    util.delete_temp_frames(temp_frame_paths[0])
            else:
                if util.has_extension(v.filename, ["gif"]):
                    skip_audio = True
                else:
                    skip_audio = moop.globals.skip_audio
                process_mgr.run_batch_inmem(
                    output_method,
                    v.filename,
                    v.finalname,
                    v.startframe,
                    v.endframe,
                    fps,
                    moop.globals.execution_threads,
                )

            if not moop.globals.processing:
                end_processing("Processing stopped!")
                return

            video_file_name = v.finalname
            if os.path.isfile(video_file_name):
                destination = ""
                if util.has_extension(v.filename, ["gif"]):
                    gifname = util.get_destfilename_from_path(
                        v.filename, moop.globals.output_path, ".gif"
                    )
                    destination = util.replace_template(gifname, index=index)
                    pathlib.Path(os.path.dirname(destination)).mkdir(
                        parents=True, exist_ok=True
                    )

                    update_status("Creating final GIF")
                    ffmpeg.create_gif_from_video(video_file_name, destination)
                    if os.path.isfile(destination):
                        os.remove(video_file_name)
                else:
                    skip_audio = moop.globals.skip_audio
                    destination = util.replace_template(video_file_name, index=index)
                    pathlib.Path(os.path.dirname(destination)).mkdir(
                        parents=True, exist_ok=True
                    )

                    if not skip_audio:
                        ffmpeg.restore_audio(
                            video_file_name,
                            v.filename,
                            v.startframe,
                            v.endframe,
                            destination,
                        )
                        if os.path.isfile(destination):
                            os.remove(video_file_name)
                    else:
                        shutil.move(video_file_name, destination)

            elif is_streaming_only == False:
                update_status(f"Failed processing {os.path.basename(v.finalname)}!")
            elapsed_time = time() - start_processing
            average_fps = (v.endframe - v.startframe) / elapsed_time
            update_status(
                f"\nProcessing {os.path.basename(destination)} took {elapsed_time:.2f} secs, {average_fps:.2f} frames/s"
            )
    end_processing("Finished")


def end_processing(msg: str):
    update_status(msg)
    moop.globals.target_folder_path = None
    release_resources()


def destroy() -> None:
    if moop.globals.target_path:
        util.clean_temp(moop.globals.target_path)
    release_resources()
    sys.exit()


def run() -> None:
    parse_args()
    if not pre_check():
        return
    moop.globals.CFG = Settings("config.yaml")
    moop.globals.cuda_device_id = moop.globals.startup_args.cuda_device_id
    moop.globals.execution_threads = moop.globals.CFG.max_threads
    moop.globals.video_encoder = moop.globals.CFG.output_video_codec
    moop.globals.video_quality = moop.globals.CFG.video_quality
    moop.globals.CFG.server_share = True
    moop.globals.max_memory = (
        moop.globals.CFG.memory_limit if moop.globals.CFG.memory_limit > 0 else None
    )
    main.run()
