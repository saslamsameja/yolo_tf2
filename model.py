import tensorflow as tf
import numpy as np

yolo_tiny_anchors = np.array([(10, 14), (23, 27), (37, 58),
                              (81, 82), (135, 169),  (344, 319)],
                             np.float32) / 416
yolo_tiny_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])
classes = 80


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    Make trainable=False freeze BN for real (the og version is sad)
    """

    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


def yolo_boxes(pred, anchors, total_classes):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1]
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, total_classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
        tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def yolo_nms(outputs, anchors, masks, total_classes):
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs
    boxes, scores, total_classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(
            scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=100,
        max_total_size=100,
        iou_threshold=0.5,
        score_threshold=0.5
    )

    return boxes, scores, total_classes, valid_detections


def yolov3_tiny(training):
    x = model_in = tf.keras.Input(shape=([416, 416, 3]), name='input')

    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding="same", use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(l=0.0005))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPool2D(2, 2, "same")(x)

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(l=0.0005))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPool2D(2, 2, "same")(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(l=0.0005))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPool2D(2, 2, "same")(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(l=0.0005))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPool2D(2, 2, "same")(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(l=0.0005))(x)
    x = BatchNormalization()(x)
    x = x_8 = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPool2D(2, 2, "same")(x)

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(l=0.0005))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.MaxPool2D(2, 1, "same")(x)

    x = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, strides=1, padding="same", use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(l=0.0005))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=1, strides=1, padding="same", use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(l=0.0005))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    out_0 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.0005))(x)
    out_0 = BatchNormalization()(out_0)
    out_0 = tf.keras.layers.LeakyReLU(alpha=0.1)(out_0)
    out_0 = tf.keras.layers.Conv2D(filters=255, kernel_size=1, strides=1, padding="same", use_bias=True,
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.0005))(out_0)
    out_0 = tf.keras.layers.Lambda(lambda val: tf.reshape(val, (-1, tf.shape(val)[1], tf.shape(val)[2],
                                                          len(yolo_tiny_anchor_masks[0]), classes+5)))(out_0)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=1, strides=1, padding="same", use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(l=0.0005))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.UpSampling2D(2)(x)
    x = tf.keras.layers.concatenate([x, x_8])

    out_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.0005))(x)
    out_1 = BatchNormalization()(out_1)
    out_1 = tf.keras.layers.LeakyReLU(alpha=0.1)(out_1)

    out_1 = tf.keras.layers.Conv2D(filters=255, kernel_size=1, strides=1, padding="same", use_bias=True,
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.0005))(out_1)

    out_1 = tf.keras.layers.Lambda(lambda val: tf.reshape(val, (-1, tf.shape(val)[1], tf.shape(val)[2],
                                                          len(yolo_tiny_anchor_masks[1]), classes+5)))(out_1)

    if training:
        return tf.keras.Model(model_in, (out_0, out_1), name='yolov3')

    boxes_0 = tf.keras.layers.Lambda(lambda val: yolo_boxes(val, yolo_tiny_anchors[yolo_tiny_anchor_masks[0]], classes),
                                     name='yolo_boxes_0')(out_0)
    boxes_1 = tf.keras.layers.Lambda(lambda val: yolo_boxes(val, yolo_tiny_anchors[yolo_tiny_anchor_masks[1]], classes),
                                     name='yolo_boxes_1')(out_1)
    outputs = tf.keras.layers.Lambda(lambda val: yolo_nms(val, yolo_tiny_anchors, yolo_tiny_anchor_masks, classes),
                                     name='yolo_nms')((boxes_0[:3], boxes_1[:3]))
    return tf.keras.Model(model_in, outputs, name='yolov3_tiny')


def get_yolo_loss(anchors, ignore_thresh=0.5):
    def yolo_loss(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(
            y_pred, anchors, classes)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(
            y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - \
            tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh),
                           tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
                x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_box, true_box, obj_mask),
            tf.float32)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # 5. calculate all losses
        xy_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = tf.losses.binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + \
            (1 - obj_mask) * ignore_mask * obj_loss
        # TODO: use binary_crossentropy instead
        class_loss = obj_mask * tf.losses.sparse_categorical_crossentropy(
            true_class_idx, pred_class)

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss
    return yolo_loss


def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
        (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
        (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)
