import torchvision
import skimage
import numpy as np
import torch
from skimage import io

normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
resize = torchvision.transforms.Resize((224,224))
transform_rgb = torchvision.transforms.Compose([
                                                torchvision.transforms.ToPILImage(),
                                                resize,
                                                torchvision.transforms.ToTensor(),
                                                normalize])

depth2rgb = lambda x: skimage.img_as_ubyte(skimage.color.gray2rgb(x/np.max(x)))
transform_depth = torchvision.transforms.Compose([
                                                depth2rgb,
                                                torchvision.transforms.ToPILImage(),
                                                resize,
                                                torchvision.transforms.ToTensor(),
                                                normalize])

def setup_device(gpu_num=0):
    """Setup device."""
    device_name = 'cuda:'+str(gpu_num) if torch.cuda.is_available() else 'cpu'  # Is there a GPU?
    device = torch.device(device_name)
    return device

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def vision_featurize(rgb, depth, gpu_num=0):
    depth_image = io.imread(depth, as_gray=True)  # cv2.imread(depth_image_loc, cv2.IMREAD_GRAYSCALE)
    rgb_image = io.imread(rgb, as_gray=False)  # cv2.imread(rgb_image_loc)
    if transform_depth:
        depth_image = transform_depth(depth_image)
    if transform_rgb:
        rgb_image = transform_rgb(rgb_image)
    device = setup_device(gpu_num=gpu_num)
    vision_model = torchvision.models.resnet152(pretrained=True)
    vision_model.fc = Identity()
    vision_model.to(device)
    vision_model.eval()
    rgb_data = vision_model(rgb_image.unsqueeze_(0).to(device)).detach().to('cpu')
    depth_data = vision_model(depth_image.unsqueeze_(0).to(device)).detach().to('cpu')
    vision_data = torch.tensor(np.concatenate((depth_data.numpy(), rgb_data.numpy()), axis=1))
    return vision_data[0]
# #feature = vq_wav2vec_featurize(wav_file="speech/train/fork_2_2_4.wav")
# feature = vision_featurize(rgb="data/gold/images/color/coffee_mug/coffee_mug_1/coffee_mug_1_1.png", depth="data/gold/images/depth/coffee_mug/coffee_mug_1/coffee_mug_1_1.png")
# print(feature)
# print(feature.shape)