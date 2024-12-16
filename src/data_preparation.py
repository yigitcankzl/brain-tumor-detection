from tensorflow.keras.preprocessing.image import ImageDataGenerator

def prepare_data(train_dir, test_dir, image_size=(150, 150), batch_size=32):
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=10,
                                       brightness_range=(0.85, 1.15),
                                       width_shift_range=0.002,
                                       height_shift_range=0.002,
                                       shear_range=12.5,
                                       zoom_range=0,
                                       horizontal_flip=True,
                                       vertical_flip=False,
                                       fill_mode="nearest")

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    return train_generator, test_generator 