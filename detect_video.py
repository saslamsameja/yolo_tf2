import time
import cv2
import numpy as np
import tensorflow as tf
from model import yolov3_tiny
from dataset import transform_images
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', help='path to classes file.',
                        default=f"./data/coco.names")
    parser.add_argument('--weights', help='The path to network weights.',
                        default=f"./checkpoints/yolov3_tiny.tf")
    parser.add_argument('--video', help='The path to video on which inference is to be called.',
                        default=f"/home/shahzaib/Downloads/getting_started/downloaded_videos/cam0/2020_1_2_18_0 .avi")
    parser.add_argument('--model_input_size', help='The size of image that needs to be passed to model for inference.',
                        default=416)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    yolo = yolov3_tiny(training=False)
    yolo.load_weights(args.weights)
    print('weights loaded')

    class_names = [c.strip() for c in open(args.classes).readlines()]
    print('classes loaded')

    times = []

    try:
        vid = cv2.VideoCapture(int(args.video))
    except:
        vid = cv2.VideoCapture(args.video)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f"./output.mp4", codec, fps, (width, height))

    while True:
        _, img = vid.read()

        if img is None:
            print("Empty Frame")
            time.sleep(0.1)
            continue

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, args.model_input_size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        t2 = time.time()
        times.append(t2-t1)
        times = times[-20:]

        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        img = cv2.putText(img, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        out.write(img)
        cv2.imshow('output', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


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
