from fastapi import APIRouter,UploadFile
import torch.nn
from .neural_network import CNN
from PIL import Image
import io
import numpy as np
from torchvision import transforms

router = APIRouter()

@router.post("/detect/") 
async def cat_dog_det(image: UploadFile):
    image_data = await image.read()
    det = get_results(image_data)
    
    response_api = 'Gato'
    if det :
        response_api = 'Perro'
    
    return {'Deteccion': response_api}


def get_results(image):

    transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    ])

    model = CNN(in_chann=3, num_class=2)
    model.load_state_dict(torch.load('model.pt', weights_only=True))
    model.eval()

    image = Image.open(io.BytesIO(image)).convert('RGB')
    tensor_image = transform(image)

    tensor_image = tensor_image.unsqueeze(0)

    with torch.no_grad():
        output = model(tensor_image)

    return int(np.array(output).argmax())