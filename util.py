import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.utils import load_img, img_to_array
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.compat.v1.logging import INFO, set_verbosity

random.seed(a=None, version=2)

set_verbosity(INFO)

def get_mean_std_per_batch(image_dir, df, H=320, W=320):
    sample_data = []
    for img_name in df.sample(100)["Image"].values:
        img_path = image_dir + img_name
        img = load_img(img_path, target_size=(H, W))
        img_array = img_to_array(img)
        sample_data.append(img_array)

    sample_data = np.array(sample_data)
    mean = np.mean(sample_data, axis=(0, 1, 2))
    std = np.std(sample_data, axis=(0, 1, 2))
    return mean, std


def load_image(img, image_dir, df, preprocess=True, H=320, W=320):
    """Load and preprocess image."""
    img_path = image_dir + img
    mean, std = get_mean_std_per_batch(image_dir, df, H=H, W=W)
    x = load_img(img_path, target_size=(H, W))
    x = img_to_array(x)  # Convert PIL image to numpy array
    if preprocess:
        x -= mean
        x /= std
        x = np.expand_dims(x, axis=0)
    return x

def grad_cam(input_model, image, cls, layer_name, H=320, W=320):
    """GradCAM method for visualizing input saliency."""
    y_c = input_model.output[0, cls]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]

    gradient_function = K.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0], grads_val[0]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # Process CAM
    cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max() if cam.max() != 0 else cam
    return cam

def compute_gradcam(model, img, image_dir, df, labels, selected_labels, layer_name='bn'):
    preprocessed_input = load_image(img, image_dir, df)
    predictions = model.predict(preprocessed_input)

    print("Loading original image")
    plt.figure(figsize=(15, 10))
    plt.subplot(151)
    plt.title("Original")
    plt.axis('off')
    original_image = load_image(img, image_dir, df, preprocess=False).astype(np.uint8)
    plt.imshow(original_image, cmap='gray')

    j = 1
    for i in range(len(labels)):
        if labels[i] in selected_labels:
            print(f"Generating gradcam for class {labels[i]}")
            gradcam = grad_cam(model, preprocessed_input, i, layer_name)
            plt.subplot(151 + j)
            plt.title(f"{labels[i]}: p={predictions[0][i]:.3f}")
            plt.axis('off')
            plt.imshow(original_image, cmap='gray')
            plt.imshow(gradcam, cmap='jet', alpha=min(0.5, predictions[0][i]))
            j += 1

    plt.tight_layout()
    plt.show()

def get_roc_curve(labels, predicted_vals, generator):
    auc_roc_vals = []
    plt.figure(figsize=(10, 10))
    for i in range(len(labels)):
        try:
            gt = generator.labels[:, i]
            pred = predicted_vals[:, i]
            auc_roc = roc_auc_score(gt, pred)
            auc_roc_vals.append(auc_roc)
            fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
            plt.plot(fpr_rf, tpr_rf, label=f"{labels[i]} (AUC = {round(auc_roc, 3)})")
        except Exception as e:
            print(f"Error generating ROC for {labels[i]}: {e}")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

    return auc_roc_vals
