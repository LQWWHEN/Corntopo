import random
import torchvision.transforms.functional as TF

class DataTransform:
    def __init__(
        self,
        hflip_prob=0.5,
        vflip_prob=0.5,
        rotation=True,
    ):
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotation = rotation
    def get_params(self, seed=None):
        """
        生成一组随机参数
        """
        rng = random.Random(seed)

        params = {
            "hflip": rng.random() < self.hflip_prob,
            "vflip": rng.random() < self.vflip_prob,
            "angle": rng.choice([0, 90, 180, 270]),
        }

        return params
    def __call__(self, x, params=None, seed=None):

        if params is None:
                params = self.get_params(seed)

        if params["hflip"]:
                x = TF.hflip(x)

        if params["vflip"]:
                x = TF.vflip(x)

        if self.rotation:
            x = TF.rotate(
                    x,
                    params["angle"],
                    interpolation=TF.InterpolationMode.BILINEAR
                )

           

      


        return x