import tensorflow as tf

import argparse
from utils import load_data
strategy = tf.distribute.MirroredStrategy()

def load_model(input_shape, alpha):
    with strategy.scope():
        full_model = tf.keras.applications.MobileNetV3Small(input_shape=input_shape, alpha=alpha, include_top=False, weights=None)
        output = tf.keras.layers.Conv2D(
            filters=1, 
            kernel_size=1, 
            strides=1, 
            padding='same',
            activation='sigmoid')(full_model.layers[166].output)
        
        model = tf.keras.Model(inputs=full_model.inputs, outputs=output)

    return model


def train(model, X_train, Y_train):
    # train
    with strategy.scope():   
        #opt = tf.keras.optimizers.Adam(lr=0.001, decay=None) 
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    callback_checkpointer = tf.keras.callbacks.ModelCheckpoint(
        'face_roi_alpha_0.1_240_15', 
        verbose=1, 
        save_best_only=True, 
        save_weights_only=False,
        monitor='val_loss',
        mode='min')

    callback_tf_logger = tf.keras.callbacks.TensorBoard(log_dir='logs_tf_logger_alpha_0.1_240_15')
    results = model.fit(X_train, Y_train, validation_split=0.2, batch_size=256, epochs=500, callbacks=[callback_checkpointer, callback_tf_logger])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_width", default=240, type=int)
    parser.add_argument("--img_height", default=240, type=int)
    parser.add_argument("--img_channels", default=3, type=int)
    parser.add_argument("--mask_height", default=15, type=int)
    parser.add_argument("--mask_width", default=15, type=int)
    parser.add_argument("--alpha", default=0.1, type=float)
    parser.add_argument("--train_data_path", type=str)

    args = parser.parse_args()

    # Load model
    model = load_model(input_shape = (args.img_width, args.img_height, args.img_channels), alpha=args.alpha)


    #Data Loading
    X_train, Y_train = load_data(
        args.train_data_path, 
        'train',
        (args.img_width, args.img_height, args.img_channels), 
        (args.mask_width, args.mask_height)
    )

    # train
    train(model, X_train, Y_train)

if __name__ == "__main__":
    main()

