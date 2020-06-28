import time
import cv2
import numpy as np
import tensorflow as tf
from model import yolov3_tiny
from dataset import transform_images
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', help='The path to weights.',
                        default=f"./checkpoints/yolov3_tiny.tf")
    parser.add_argument('--classes', help='The path to file containing classes names.',
                        default=f"./data/coco.names")
    parser.add_argument('--image', help='The path to image on which inference is to be called.',
                        default=f"./data/street.jpg")
    parser.add_argument('--model_input_size', help='The size of image that needs to be passed to model for inference.',
                        default=416)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    yolo = yolov3_tiny(training=False)

    yolo.load_weights(args.weights).expect_partial()
    print('weights loaded')

    class_names = [c.strip() for c in open(args.classes).readlines()]
    print('classes loaded')

    img_raw = tf.image.decode_image(open(args.image, 'rb').read(), channels=3)

    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, args.model_input_size)

    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
    print('time: {}'.format(t2 - t1))

    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    cv2.imwrite(f"./output.jpg", img)


def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
            x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img


if __name__ == '__main__':
    main()
