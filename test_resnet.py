import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Load your trained model
model = tf.keras.models.load_model('resnet50_detection_model.h5')

# 2. Prepare your labeled test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'C:/Users/mysel/PycharmProjects/FYP2/fire_dataset/test',  # Folder containing categorized subfolders
    target_size=(224, 224),       # Ensure this matches what the model expects
    batch_size=32,
    class_mode='categorical',     # Or 'binary' depending on your setup
    shuffle=False
)

# 3. Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f"ResNet50 Detection Model Accuracy: {accuracy * 100:.2f}%")