import argparse
import cv2
import numpy as np

import oneflow as flow
from flowvision.models import ModelCreator


def load_image(image_path):
    im = cv2.imread(image_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = np.transpose(im, (2, 0, 1))
    im = np.expand_dims(im, axis=0)
    return np.ascontiguousarray(im, "float32")


def recover_image(im):
    im = np.squeeze(im)
    im = np.transpose(im, (1, 2, 0))
    im = cv2.cvtColor(np.float32(im), cv2.COLOR_RGB2BGR)
    return im.astype(np.uint8)


def stylize(args):
    content_image = load_image(args.content_image)
    style_model = ModelCreator.create_model("neural_style_transfer", pretrained=True, style_model=args.style_model)
    with flow.no_grad():
        style_model.to("cuda")
        output = style_model(flow.Tensor(content_image).clamp(0, 255).to("cuda"))
    cv2.imwrite(args.output_image, recover_image(output.numpy()))


def main():
    arg_parser = argparse.ArgumentParser(
        description="parser for fast-neural-style"
    )
    arg_parser.add_argument(
        "--content-image",
        type=str,
        required=True,
        help="path to content image you want to stylize",
    )
    arg_parser.add_argument(
        "--style-model",
        type=str,
        required=True,
        default="sketch",
        help="path to content image you want to stylize",
    )
    arg_parser.add_argument(
        "--output-image",
        type=str,
        required=True,
        help="path for saving the output image",
    )

    args = arg_parser.parse_args()

    stylize(args)


if __name__ == "__main__":
    main()
