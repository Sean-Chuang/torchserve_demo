import io
import os
import logging
import torch
import numpy as np

from PIL import Image
from torch.autograd import Variable
from torchvision import transforms


logger = logging.getLogger(__name__)


class MNISTDigitClassifier(object):
    """
    MNISTDigitClassifier handler class. This handler takes a greyscale image
    and returns the digit in that image.
    """

    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False

    def initialize(self, ctx):
        """First try to load torchscript else load eager mode state_dict based model"""
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.init_model(model_dir, properties.get("gpu_id"))


    def init_model(self, model_dir, gpu_id=0):
        self.device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        model_pt_path = os.path.join(model_dir, "model.pt")
        # Read model definition file
        model_def_path = os.path.join(model_dir, "model.py")
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model definition file")

        from model import Net
        state_dict = torch.load(model_pt_path, map_location=self.device)
        self.model = Net()
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.debug('Model file {0} loaded successfully'.format(model_pt_path))
        self.initialized = True


    def preprocess(self, data):
        """
         Scales, crops, and normalizes a PIL image for a MNIST model,
         returns an Numpy array
        """
        image = data[0].get("data")
        if image is None:
            image = data[0].get("body")

        mnist_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        image = Image.open(io.BytesIO(image))
        image = mnist_transform(image)
        return image

    def inference(self, img, topk=5):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        # Convert 2D image to 1D vector
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img)

        self.model.eval()
        inputs = Variable(img).to(self.device)
        outputs = self.model.forward(inputs)
        _, y_hat = outputs.max(1)
        predicted_idx = str(y_hat.item())
        return [predicted_idx]

    def postprocess(self, inference_output):
        return inference_output


_service = MNISTDigitClassifier()

def init(context):
    if context:
        _service.initialize(context)
    else:
        model_dir = "model"
        _service.init_model(model_dir)


def handle(data, context=None):
    if not _service.initialized:
        init(context)

    if data is None:
        return None

    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    return data
