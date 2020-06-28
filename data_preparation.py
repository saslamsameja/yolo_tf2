import tensorflow as tf
import tqdm
import cv2
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', help='The path to the video.',
                        default=f"./video.avi")
    parser.add_argument('--ann_path', help='The path to annotations of the video.',
                        default=f"./annotation.txt")
    parser.add_argument('--out_file', help='The path to output tf record file.',
                        default=f"./data/train_data.tfrecord")

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    writer = tf.io.TFRecordWriter(args.out_file)
    text_data = open(args.ann_path).readlines()
    cap = cv2.VideoCapture(args.video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for index in tqdm.tqdm(range(total_frames)):
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        ret, frame = cap.read()
        if not ret:
            continue

        img_raw = tf.image.encode_jpeg(frame).numpy()

        data = [x.split(",")[-5:-1] for x in text_data[index].split("person")][:-1]

        for data_i in data:
            tl = (int(data_i[0]), int(data_i[1]))
            br = (tl[0]+int(data_i[2]), tl[1]+int(data_i[3]))
            xmin.append(tl[0]/frame.shape[1])
            ymin.append(tl[1]/frame.shape[0])
            xmax.append(br[0]/frame.shape[1])
            ymax.append(br[1]/frame.shape[0])

        classes_text = [b'person']*len(data)
        tf_example = tf.train.Example(features=tf.train.Features(feature={
                'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
                'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
                'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
                'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
                'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
            }))
        writer.write(tf_example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    main()
