import numpy as np
import cv2


class Vignetting(object):
    """
    Applies a vignetting effect to an image.
    """
    def __init__(self,
                 ratio_min_dist=0.2,
                 range_vignette=(0.2, 0.8),
                 random_sign=False):
        self.ratio_min_dist = ratio_min_dist
        self.range_vignette = np.array(range_vignette)
        self.random_sign = random_sign

    def __call__(self, X, Y=None):
        # If image is in RGBA format, separate the RGB and alpha channels
        if X.shape[2] == 4:
            X_rgb = X[..., :3]
            X_a = X[..., 3:4]
        else:
            X_rgb = X

        h, w = X_rgb.shape[:2]
        min_dist = np.array([h, w]) / 2 * np.random.random() * self.ratio_min_dist

        # create matrix of distance from the center on the two axis
        x, y = np.meshgrid(np.linspace(-w / 2, w / 2, w), np.linspace(-h / 2, h / 2, h))
        x, y = np.abs(x), np.abs(y)

        # create the vignette mask on the two axis
        x = (x - min_dist[0]) / (np.max(x) - min_dist[0])
        x = np.clip(x, 0, 1)
        y = (y - min_dist[1]) / (np.max(y) - min_dist[1])
        y = np.clip(y, 0, 1)

        # then get a random intensity of the vignette
        vignette = (x + y) / 2 * np.random.uniform(*self.range_vignette)
        vignette = np.tile(vignette[..., None], [1, 1, 3])

        sign = 2 * (np.random.random() < 0.5) * (self.random_sign) - 1
        X_rgb = X_rgb * (1 + sign * vignette)

        # If original image was in RGBA format, combine RGB with alpha channel
        if X.shape[2] == 4:
            X = np.concatenate([X_rgb, X_a], axis=-1)
        else:
            X = X_rgb

        return X


class LensDistortion(object):
    """
    Applies a lens distortion effect to an image.
    """
    def __init__(self, d_coef=(0.15, 0.05, 0.1, 0.1, 0.05)):
        self.d_coef = np.array(d_coef)

    def __call__(self, X):
        # get the height and the width of the image
        h, w = X.shape[:2]

        # compute its diagonal
        f = (h ** 2 + w ** 2) ** 0.5

        # set the image projective to carrtesian dimension
        K = np.array([[f, 0, w / 2],
                      [0, f, h / 2],
                      [0, 0, 1]])

        d_coef = self.d_coef * np.random.random(5)  # value
        d_coef = d_coef * (2 * (np.random.random(5) < 0.5) - 1)  # sign
        # Generate new camera matrix from parameters
        M, _ = cv2.getOptimalNewCameraMatrix(K, d_coef, (w, h), 0)

        # Generate look-up tables for remapping the camera image
        remap = cv2.initUndistortRectifyMap(K, d_coef, None, M, (w, h), 5)

        # Remap the original image to a new image
        X = cv2.remap(X, *remap, cv2.INTER_LINEAR)
        return X


class UniformNoise(object):
    """
    Adds uniform noise to an image.
    """
    def __init__(self, low=-50, high=50):
        self.low = low
        self.high = high

    def __call__(self, X):
        noise = np.random.uniform(self.low, self.high, X.shape)
        X = X + noise
        return X


if __name__ == "__main__":
    from PIL import Image

    # Load an image
    img = Image.open("./utils/test/test10.png")  # relative path from the root of application
    img = np.array(img)

    # Create an instance of each augmentation class
    vignet = Vignetting()
    un = UniformNoise()
    ld = LensDistortion()

    # Apply each augmentation to the image
    img_vignet = vignet(img)
    img_un = un(img)
    img_ld = ld(img)

    # Convert the arrays to uint8 and scale the pixel values to the range [0, 255]
    img_vignet = (img_vignet * 255).astype(np.uint8)
    img_un = (img_un * 255).astype(np.uint8)
    img_ld = (img_ld * 255).astype(np.uint8)

    # Print the type and shape of each image
    print("Vignet image type:", type(img_vignet), "shape:", img_vignet.shape)
    print("Uniform noise image type:", type(img_un), "shape:", img_un.shape)
    print("Lens distortion image type:", type(img_ld), "shape:", img_ld.shape)
