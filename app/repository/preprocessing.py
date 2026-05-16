import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
import torchvision.transforms.functional as F
from app.repository.schema import ModelInput
import torchvision.transforms as transforms

class Preprocessing():
    @staticmethod
    def load_and_validate_image(file) -> Image.Image:
        try:
            image = Image.open(file)
            image.load()

            return image.convert('RGB')
        except (UnidentifiedImageError, ValueError, TypeError) as e:
            raise ValueError(f"The uploaded file is not a valid image. Details: {e}")


    @staticmethod
    def resize(image: Image.Image, size=(512,512), resampling_mode='BILINEAR') -> Image.Image:
        if resampling_mode == 'BILINEAR':
            return image.resize(size, resample=Image.Resampling.BILINEAR)
        elif resampling_mode == 'NEAREST':
            return image.resize(size, resample=Image.Resampling.NEAREST)
        else:
            raise ValueError(f"Invalid resampling mode " + resampling_mode)


    @staticmethod
    def normalize(image_tensor: torch.Tensor) -> torch.Tensor:
        normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )

        #normalized_tensor = F.normalize(image_tensor,mean=mean, std=std)
        return normalize(image_tensor)

    @staticmethod
    def build_model_input(pre=None, post=None, premask=None) -> ModelInput:
        post_img_raw = Preprocessing.load_and_validate_image(post)
        post_img = Preprocessing.resize(post_img_raw)
        tensor_post = Preprocessing.normalize(transforms.ToTensor()(post_img)).unsqueeze(0)

        if pre is not None:
            pre_img_raw = Preprocessing.load_and_validate_image(pre)
            pre_img = Preprocessing.resize(pre_img_raw)
            tensor_pre = Preprocessing.normalize(transforms.ToTensor()(pre_img)).unsqueeze(0)
        else:
            tensor_pre = None

        if premask is not None:
            premask_img_raw = Image.open(premask).convert('L')
            premask_img = Preprocessing.resize(premask_img_raw,resampling_mode='NEAREST')
            premask_np = np.array(premask_img)
            tensor_premask = torch.from_numpy(premask_np).float().unsqueeze(0)
            if tensor_premask.max() > 1.0:
                tensor_premask = tensor_premask / 255.0
            tensor_premask = tensor_premask.unsqueeze(0)
        else:
            tensor_premask = None


        return ModelInput(
            pre_image = tensor_pre,
            post_image = tensor_post,
            pre_mask = tensor_premask
        )






