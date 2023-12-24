import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_custom_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_custom_nodes()


from nodes import (
    LoraLoaderModelOnly,
    EmptyLatentImage,
    VAELoader,
    NODE_CLASS_MAPPINGS,
    CLIPVisionLoader,
    LoadImage,
    SaveImage,
    CheckpointLoaderSimple,
    VAEDecode,
    CLIPTextEncode,
    KSampler,
)


def main():
    import_custom_nodes()
    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_20 = checkpointloadersimple.load_checkpoint(
            ckpt_name="SD1.5/beautifulRealistic_v7.safetensors"
        )

        loraloadermodelonly = LoraLoaderModelOnly()
        loraloadermodelonly_31 = loraloadermodelonly.load_lora_model_only(
            lora_name="SD1.5/ip-adapter-faceid_sd15_lora.safetensors",
            strength_model=1.0,
            model=get_value_at_index(checkpointloadersimple_20, 0),
        )

        insightfaceloader = NODE_CLASS_MAPPINGS["InsightFaceLoader"]()
        insightfaceloader_32 = insightfaceloader.load_insight_face(provider="CPU")

        ipadaptermodelloader = NODE_CLASS_MAPPINGS["IPAdapterModelLoader"]()
        ipadaptermodelloader_34 = ipadaptermodelloader.load_ipadapter_model(
            ipadapter_file="ip-adapter-faceid_sd15.bin"
        )

        loadimage = LoadImage()
        loadimage_36 = loadimage.load_image(
            image="Image 2023-12-23 at 8.12 PM (1).jpeg"
        )

        ipadaptermodelloader_37 = ipadaptermodelloader.load_ipadapter_model(
            ipadapter_file="ip-adapter-plus-face_sd15.safetensors"
        )

        clipvisionloader = CLIPVisionLoader()
        clipvisionloader_38 = clipvisionloader.load_clip(
            clip_name="SD1.5/pytorch_model.bin"
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_41 = cliptextencode.encode(
            text="photo of a man, fashion model, simple clothes\n\nhigh quality, highly detailed, 4k, highres",
            clip=get_value_at_index(checkpointloadersimple_20, 1),
        )

        cliptextencode_42 = cliptextencode.encode(
            text="blurry, distorted, low quality, bad hands",
            clip=get_value_at_index(checkpointloadersimple_20, 1),
        )

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_44 = emptylatentimage.generate(
            width=512, height=768, batch_size=4
        )

        vaeloader = VAELoader()
        vaeloader_46 = vaeloader.load_vae(
            vae_name="vae-ft-mse-840000-ema-pruned.safetensors"
        )

        seed_rgthree = NODE_CLASS_MAPPINGS["Seed (rgthree)"]()
        seed_rgthree_49 = seed_rgthree.main(seed=random.randint(1, 2**64))

        facerestoremodelloader = NODE_CLASS_MAPPINGS["FaceRestoreModelLoader"]()
        facerestoremodelloader_54 = facerestoremodelloader.load_model(
            model_name="GFPGANv1.4.pth"
        )

        load_face_analysis_model_mtb = NODE_CLASS_MAPPINGS[
            "Load Face Analysis Model (mtb)"
        ]()
        load_face_analysis_model_mtb_59 = load_face_analysis_model_mtb.load_model(
            faceswap_model="buffalo_l"
        )

        load_face_swap_model_mtb = NODE_CLASS_MAPPINGS["Load Face Swap Model (mtb)"]()
        load_face_swap_model_mtb_60 = load_face_swap_model_mtb.load_model(
            faceswap_model="inswapper_128.onnx"
        )

        upscalemodelloader = NODE_CLASS_MAPPINGS["UpscaleModelLoader"]()
        upscalemodelloader_66 = upscalemodelloader.load_model(
            model_name="4x-UltraSharp.pth"
        )

        ipadapterapply = NODE_CLASS_MAPPINGS["IPAdapterApply"]()
        prepimageforclipvision = NODE_CLASS_MAPPINGS["PrepImageForClipVision"]()
        ksampler = KSampler()
        vaedecode = VAEDecode()
        facerestorecfwithmodel = NODE_CLASS_MAPPINGS["FaceRestoreCFWithModel"]()
        face_swap_mtb = NODE_CLASS_MAPPINGS["Face Swap (mtb)"]()
        imageupscalewithmodel = NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]()
        saveimage = SaveImage()

        for q in range(5):
            ipadapterapply_35 = ipadapterapply.apply_ipadapter(
                weight=0.9,
                noise=0.0,
                weight_type="original",
                start_at=0.0,
                end_at=1.0,
                unfold_batch=False,
                ipadapter=get_value_at_index(ipadaptermodelloader_34, 0),
                clip_vision=get_value_at_index(insightfaceloader_32, 0),
                image=get_value_at_index(loadimage_36, 0),
                model=get_value_at_index(loraloadermodelonly_31, 0),
            )

            prepimageforclipvision_39 = prepimageforclipvision.prep_image(
                interpolation="LANCZOS",
                crop_position="center",
                sharpening=0.0,
                image=get_value_at_index(loadimage_36, 0),
            )

            ipadapterapply_40 = ipadapterapply.apply_ipadapter(
                weight=0.3,
                noise=0.33,
                weight_type="original",
                start_at=0.0,
                end_at=1.0,
                unfold_batch=False,
                ipadapter=get_value_at_index(ipadaptermodelloader_37, 0),
                clip_vision=get_value_at_index(clipvisionloader_38, 0),
                image=get_value_at_index(prepimageforclipvision_39, 0),
                model=get_value_at_index(ipadapterapply_35, 0),
            )

            ksampler_43 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=30,
                cfg=8.0,
                sampler_name="euler",
                scheduler="normal",
                denoise=1.0,
                model=get_value_at_index(ipadapterapply_40, 0),
                positive=get_value_at_index(cliptextencode_41, 0),
                negative=get_value_at_index(cliptextencode_42, 0),
                latent_image=get_value_at_index(emptylatentimage_44, 0),
            )

            vaedecode_45 = vaedecode.decode(
                samples=get_value_at_index(ksampler_43, 0),
                vae=get_value_at_index(vaeloader_46, 0),
            )

            facerestorecfwithmodel_55 = facerestorecfwithmodel.restore_face(
                facedetection="retinaface_resnet50",
                codeformer_fidelity=0.5,
                facerestore_model=get_value_at_index(facerestoremodelloader_54, 0),
                image=get_value_at_index(vaedecode_45, 0),
            )

            face_swap_mtb_57 = face_swap_mtb.swap(
                faces_index="0",
                image=get_value_at_index(vaedecode_45, 0),
                reference=get_value_at_index(loadimage_36, 0),
                faceanalysis_model=get_value_at_index(
                    load_face_analysis_model_mtb_59, 0
                ),
                faceswap_model=get_value_at_index(load_face_swap_model_mtb_60, 0),
            )

            facerestorecfwithmodel_61 = facerestorecfwithmodel.restore_face(
                facedetection="retinaface_resnet50",
                codeformer_fidelity=0.5,
                facerestore_model=get_value_at_index(facerestoremodelloader_54, 0),
                image=get_value_at_index(face_swap_mtb_57, 0),
            )

            imageupscalewithmodel_65 = imageupscalewithmodel.upscale(
                upscale_model=get_value_at_index(upscalemodelloader_66, 0),
                image=get_value_at_index(facerestorecfwithmodel_61, 0),
            )

            saveimage_71 = saveimage.save_images(
                filename_prefix="ipadapter/faceid-upscale-api",
                images=get_value_at_index(imageupscalewithmodel_65, 0),
            )


if __name__ == "__main__":
    main()
