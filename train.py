import tensorflow as tf
import os
import dataset
import numpy as np
from model import yolov3_tiny, yolo_tiny_anchors, yolo_tiny_anchor_masks, get_yolo_loss
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', help='The path to training dataset.',
                        default=f"./data/train_data.tfrecord")
    parser.add_argument('--val_dataset', help='The path to validation dataset.',
                        default=f"./data/val_data.tfrecord")
    parser.add_argument('--weights', help='The path to weights checkpoint.',
                        default=f"./checkpoints/yolov3_tiny.tf")
    parser.add_argument('--classes', help='The path to classes file.',
                        default=f"./data/coco.names")
    parser.add_argument('--size', help='The model input image size.',
                        default=416)
    parser.add_argument('--epochs', help='The number of epochs for training.',
                        default=4)
    parser.add_argument('--batch_size', help='The batch size for training.',
                        default=8)
    parser.add_argument('--learning_rate', help='learning_rate.',
                        default=0.001)
    parser.add_argument('--num_classes', help='Number of classes in the model.',
                        default=8)
    parser.add_argument('--mode', help='fit: model.fit, eager_fit: model.fit(run_eagerly=True), eager_tf: custom GradientTape',
                        default='eager_tf')
    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    model = yolov3_tiny(training=True)
    anchors = yolo_tiny_anchors
    anchor_masks = yolo_tiny_anchor_masks

    train_dataset = dataset.load_tfrecord_dataset(args.train_dataset, args.classes, args.size)
    train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(args.batch_size)
    train_dataset = train_dataset.map(lambda x, y: (dataset.transform_images(x, args.size),
                                                    dataset.transform_targets(y, anchors, anchor_masks, args.size)))
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = dataset.load_tfrecord_dataset(args.val_dataset, args.classes, args.size)
    val_dataset = val_dataset.batch(args.batch_size)
    val_dataset = val_dataset.map(lambda x, y: (dataset.transform_images(x, args.size),
                                                dataset.transform_targets(y, anchors, anchor_masks, args.size)))
    # All other transfer require matching classes
    model.load_weights(args.weights)

    optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate)
    loss = [get_yolo_loss(anchors[mask]) for mask in anchor_masks]

    if args.mode == 'eager_tf':
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

        for epoch in range(1, args.epochs + 1):
            for batch, (images, labels) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    outputs = model(images, training=True)
                    regularization_loss = tf.reduce_sum(model.losses)
                    pred_loss = []
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))

                print(f"{epoch}_train_{batch}, {total_loss.numpy()}, {list(map(lambda x: np.sum(x.numpy()), pred_loss))}")
                avg_loss.update_state(total_loss)

            for batch, (images, labels) in enumerate(val_dataset):
                outputs = model(images)
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                for output, label, loss_fn in zip(outputs, labels, loss):
                    pred_loss.append(loss_fn(label, output))
                total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                print(f"{epoch}_val_{batch}, {total_loss.numpy()}, {list(map(lambda x: np.sum(x.numpy()), pred_loss))}")
                avg_val_loss.update_state(total_loss)

            print(f"{epoch}, train: {avg_loss.result().numpy()}, val: {avg_val_loss.result().numpy()}")
            avg_loss.reset_states()
            avg_val_loss.reset_states()
            model.save_weights('checkpoints/yolov3_train_{}.tf'.format(epoch))
    else:
        model.compile(optimizer=optimizer, loss=loss,
                      run_eagerly=(args.mode == 'eager_fit'))

        if not os.path.exists(f"./logs"):
            os.mkdir(f"logs")

        callbacks = [tf.keras.callbacks.ReduceLROnPlateau(verbose=1), tf.keras.callbacks.EarlyStopping(patience=3, verbose=1),
                     tf.keras.callbacks.ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf', verbose=1, save_weights_only=True),
                     tf.keras.callbacks.TensorBoard(log_dir='logs')]

        history = model.fit(train_dataset,
                            epochs=args.epochs,
                            callbacks=callbacks,
                            validation_data=val_dataset)
        _ = history


if __name__ == '__main__':
    main()
