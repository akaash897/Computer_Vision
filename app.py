import streamlit as st
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, LeakyReLU, BatchNormalization, Input #type:ignore
from tensorflow.keras.models import Model #type:ignore
import time
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pywt
from scipy.signal import wiener
from PIL import Image
import io

# Define constants
DATA_DIR = "dataset"
DATASET_PATH = os.path.join(DATA_DIR, "UCMerced_LandUse", "Images")
SELECTED_CATEGORIES = ['agricultural', 'baseballdiamond', 'beach', 'buildings', 'forest',
                       'airplane', 'freeway', 'golfcourse',
                       'harbor', 'mobilehomepark']
HR_SIZE = 256
SCALE_FACTOR = 2
LR_SIZE = HR_SIZE // SCALE_FACTOR

# Set page config
st.set_page_config(
    page_title="Image Enhancement App",
    page_icon="🖼️",
    layout="wide"
)

# Helper classes and functions
class InterpolationMethods:
    """Traditional interpolation methods for image upscaling"""

    @staticmethod
    def bicubic(image, scale=2):
        """Apply bicubic interpolation to upscale the image"""
        h, w = image.shape[:2]
        return cv2.resize(image, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def lanczos(image, scale=2):
        """Apply Lanczos interpolation to upscale the image"""
        h, w = image.shape[:2]
        return cv2.resize(image, (w*scale, h*scale), interpolation=cv2.INTER_LANCZOS4)

class SuperResolutionModel:
    """Deep learning-based super-resolution model"""
    @staticmethod
    def build_generator(scale_factor=1):
        """Build a super-resolution generator model based on CNN architecture"""
        input_layer = Input(shape=(None, None, 3))

        x = Conv2D(64, (3, 3), padding="same")(input_layer)
        x = LeakyReLU(alpha=0.2)(x)

        for _ in range(8):
            skip = x
            x = Conv2D(64, (3, 3), padding="same")(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = Conv2D(64, (3, 3), padding="same")(x)
            x = BatchNormalization()(x)
            x = tf.keras.layers.add([x, skip])

        if scale_factor > 1:
            for _ in range(int(np.log2(scale_factor))):
                x = UpSampling2D(size=2)(x)
                x = Conv2D(32, (3, 3), padding="same")(x)
                x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(3, (3, 3), padding="same", activation="sigmoid")(x)

        model = Model(input_layer, x)
        return model

class EnhancementMethods:
    """Image enhancement techniques"""

    @staticmethod
    def wavelet_sharpening(image):
        """Apply wavelet-based sharpening to enhance image details"""
        img_float = image.astype(np.float32)

        try:
            if len(img_float.shape) == 3:
                r, g, b = cv2.split(img_float)
                result_channels = []
                
                for channel in [r, g, b]:
                    try:
                        coeffs = pywt.dwt2(channel, 'haar')
                        # Apply safety check before multiplication
                        detail_coeffs = list(coeffs[1])
                        for i in range(len(detail_coeffs)):
                            detail_coeffs[i] = np.nan_to_num(detail_coeffs[i] * 1.5, nan=0.0)
                        
                        sharp_channel = pywt.idwt2((coeffs[0], tuple(detail_coeffs)), 'haar')
                        # Handle potential size mismatch
                        if sharp_channel.shape != channel.shape:
                            sharp_channel = cv2.resize(sharp_channel, (channel.shape[1], channel.shape[0]))
                        
                        result_channels.append(sharp_channel)
                    except Exception as e:
                        # Fallback if wavelet transform fails
                        result_channels.append(channel)
                
                sharpened = cv2.merge(result_channels)
            else:
                try:
                    coeffs = pywt.dwt2(img_float, 'haar')
                    detail_coeffs = list(coeffs[1])
                    for i in range(len(detail_coeffs)):
                        detail_coeffs[i] = np.nan_to_num(detail_coeffs[i] * 1.5, nan=0.0)
                    
                    sharpened = pywt.idwt2((coeffs[0], tuple(detail_coeffs)), 'haar')
                    # Handle potential size mismatch
                    if sharpened.shape != img_float.shape:
                        sharpened = cv2.resize(sharpened, (img_float.shape[1], img_float.shape[0]))
                except Exception as e:
                    # Fallback if wavelet transform fails
                    sharpened = img_float

            # Final safety check for invalid values
            sharpened = np.nan_to_num(sharpened, nan=0.0, posinf=255.0, neginf=0.0)
            return np.clip(sharpened, 0, 255).astype(np.uint8)
            
        except Exception as e:
            # Ultimate fallback
            return image

    @staticmethod
    def wiener_filter(image, kernel_size=(5, 5)):
        def safe_wiener(channel):
            # Ensure float32 for processing
            channel = channel.astype(np.float32)
            # Apply Wiener filter
            filtered = wiener(channel, kernel_size)
            # Replace NaNs and Infs with zeros or safe values
            filtered = np.nan_to_num(filtered, nan=0.0, posinf=255.0, neginf=0.0)
            # Clip and convert back
            return np.clip(filtered, 0, 255).astype(np.uint8)

        if len(image.shape) == 3:
            r, g, b = cv2.split(image)
            r_deblurred = safe_wiener(r)
            g_deblurred = safe_wiener(g)
            b_deblurred = safe_wiener(b)
            deblurred = cv2.merge([r_deblurred, g_deblurred, b_deblurred])
        else:
            deblurred = safe_wiener(image)

        return deblurred

    @staticmethod
    def wavelet_sr_hybrid(sr_image):
        """Combine Super-Resolution with Wavelet Sharpening"""
        img_uint8 = (sr_image * 255).astype(np.uint8)
        sharpened = EnhancementMethods.wavelet_sharpening(img_uint8)
        return sharpened.astype(np.float32) / 255.0

    @staticmethod
    def wiener_sr_hybrid(sr_image):
        """Combine Super-Resolution with Wiener Filter"""
        img_uint8 = (sr_image * 255).astype(np.uint8)
        deblurred = EnhancementMethods.wiener_filter(img_uint8)
        return deblurred.astype(np.float32) / 255.0

class EvaluationMetrics:
    """Methods to evaluate image enhancement quality"""

    @staticmethod
    def calculate_psnr(img1, img2):
        """Calculate Peak Signal-to-Noise Ratio"""
        # Ensure both images have the same dtype
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        return psnr(img1, img2, data_range=1.0)

    @staticmethod
    def calculate_ssim(img1, img2):
        """Calculate Structural Similarity Index"""
        return ssim(img1, img2, data_range=1.0, channel_axis=-1)

    @staticmethod
    def calculate_mse(img1, img2):
        """Calculate Mean Squared Error"""
        return np.mean((img1 - img2) ** 2)

def verify_dataset_structure():
    """Verify and adjust the dataset path if needed"""
    global DATASET_PATH
    
    # Check if the dataset path exists
    if not os.path.exists(DATASET_PATH):
        st.warning(f"Default dataset path not found: {DATASET_PATH}")
        
        # Try to find the correct path by looking for the Images directory
        for root, dirs, files in os.walk(DATA_DIR):
            if "Images" in dirs:
                DATASET_PATH = os.path.join(root, "Images")
                st.success(f"Found images at: {DATASET_PATH}")
                return True
                
        # If still not found, look for any of our categories
        for category in SELECTED_CATEGORIES:
            for root, dirs, files in os.walk(DATA_DIR):
                if category in dirs:
                    DATASET_PATH = root
                    st.success(f"Found categories at: {DATASET_PATH}")
                    return True
                    
        st.error("Could not find the dataset structure. Please ensure the UC Merced Land Use dataset exists in the 'dataset' folder.")
        return False
    
    return True

def get_available_categories():
    """Get available categories from the dataset"""
    available_categories = []
    
    if not os.path.exists(DATASET_PATH):
        st.error(f"Dataset path not found: {DATASET_PATH}")
        return []
        
    # Get all directories in the dataset path
    try:
        directories = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
        
        # Filter to only include selected categories that exist
        for category in SELECTED_CATEGORIES:
            if category in directories:
                available_categories.append(category)
                
        if not available_categories:
            st.warning("No matching categories found in the dataset.")
            # Try to show what's available instead
            st.info(f"Available directories: {', '.join(directories[:10])}")
            
    except Exception as e:
        st.error(f"Error getting categories: {str(e)}")
        
    return available_categories

def load_single_image(image_path):
    """Load a single image for processing"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            st.error(f"Could not read image: {image_path}")
            return None, None, None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hr_img = cv2.resize(img, (HR_SIZE, HR_SIZE))
        lr_img = cv2.resize(img, (LR_SIZE, LR_SIZE))
        lr_img = cv2.GaussianBlur(lr_img, (3, 3), 0.5)
        
        return img, hr_img / 255.0, lr_img / 255.0
    
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None, None, None

def process_image(lr_img, method_name, model=None):
    """Process the image with the selected enhancement method"""
    if lr_img is None:
        st.error("No input image provided for processing")
        return None
        
    try:
        result = None
        
        if method_name == "Bicubic":
            result = InterpolationMethods.bicubic(lr_img, scale=SCALE_FACTOR)
        elif method_name == "Lanczos":
            result = InterpolationMethods.lanczos(lr_img, scale=SCALE_FACTOR)
        elif method_name == "CNN-SR":
            if model is None:
                st.error("Super-Resolution model not loaded")
                return None
            
            # Ensure the image is in the right shape for the model
            if len(lr_img.shape) == 3:
                lr_batch = np.expand_dims(lr_img, axis=0)
            else:
                st.error("Invalid image shape")
                return None
                
            sr_img = model.predict(lr_batch)[0]
            result = sr_img
        elif method_name == "Wavelet":
            upscaled = cv2.resize(lr_img, (HR_SIZE, HR_SIZE))
            img_uint8 = (upscaled * 255).astype(np.uint8)
            enhanced = EnhancementMethods.wavelet_sharpening(img_uint8)
            result = enhanced.astype(np.float32) / 255.0
        elif method_name == "Wiener":
            upscaled = cv2.resize(lr_img, (HR_SIZE, HR_SIZE))
            img_uint8 = (upscaled * 255).astype(np.uint8)
            enhanced = EnhancementMethods.wiener_filter(img_uint8)
            result = enhanced.astype(np.float32) / 255.0
        elif method_name == "Wavelet-SR":
            if model is None:
                st.error("Super-Resolution model not loaded")
                return None
            
            # First get the SR result
            lr_batch = np.expand_dims(lr_img, axis=0)
            sr_img = model.predict(lr_batch)[0]
            result = EnhancementMethods.wavelet_sr_hybrid(sr_img)
        elif method_name == "Wiener-SR":
            if model is None:
                st.error("Super-Resolution model not loaded")
                return None
            
            # First get the SR result
            lr_batch = np.expand_dims(lr_img, axis=0)
            sr_img = model.predict(lr_batch)[0]
            result = EnhancementMethods.wiener_sr_hybrid(sr_img)
        else:
            st.error(f"Unknown method: {method_name}")
            return None
        
        # Ensure the result is properly clipped to valid range
        if result is not None:
            # Check for NaN or Inf values
            if np.isnan(result).any() or np.isinf(result).any():
                st.warning(f"Method {method_name} produced NaN or Inf values. Fixing...")
                result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Ensure values are in [0,1] range
            result = np.clip(result, 0, 1)
            
        return result
    
    except Exception as e:
        st.error(f"Error processing image with {method_name}: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

def calculate_metrics(enhanced_img, hr_img):
    """Calculate image quality metrics"""
    try:
        # Ensure both images have the same shape
        if enhanced_img.shape != hr_img.shape:
            st.warning("Images have different shapes. Resizing for comparison.")
            enhanced_img = cv2.resize(enhanced_img, (hr_img.shape[1], hr_img.shape[0]))
        
        # Ensure both images have the same datatype
        enhanced_img = enhanced_img.astype(np.float32)
        hr_img = hr_img.astype(np.float32)
        
        # Check for NaN or Inf values
        if np.isnan(enhanced_img).any() or np.isnan(hr_img).any() or np.isinf(enhanced_img).any() or np.isinf(hr_img).any():
            st.warning("Images contain NaN or Inf values. Fixing for metrics calculation.")
            enhanced_img = np.nan_to_num(enhanced_img, nan=0.0, posinf=1.0, neginf=0.0)
            hr_img = np.nan_to_num(hr_img, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Ensure values are in [0,1] range
        enhanced_img = np.clip(enhanced_img, 0, 1)
        hr_img = np.clip(hr_img, 0, 1)
        
        # Calculate metrics
        ssim_val = EvaluationMetrics.calculate_ssim(enhanced_img, hr_img)
        psnr_val = EvaluationMetrics.calculate_psnr(enhanced_img, hr_img)
        mse_val = EvaluationMetrics.calculate_mse(enhanced_img, hr_img)
        
        return ssim_val, psnr_val, mse_val
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return 0, 0, 0

def convert_to_displayable(image):
    """Convert image to format suitable for display"""
    try:
        if image is None:
            return None
            
        # First ensure we're working with a numpy array
        if not isinstance(image, np.ndarray):
            st.error("Image is not a numpy array")
            return None
        
        # Check if image has NaN or inf values
        if np.isnan(image).any() or np.isinf(image).any():
            st.warning("Image contains NaN or inf values. Fixing...")
            image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Convert to correct format depending on range
        if image.max() <= 1.0 and image.min() >= 0.0:
            # Already in [0,1] range, convert to uint8
            image = (image * 255).astype(np.uint8)
        elif image.max() <= 255.0 and image.min() >= 0.0:
            # Already in [0,255] range, just convert to uint8
            image = image.astype(np.uint8)
        else:
            # Normalize to [0,1] range, then convert to uint8
            image = image - image.min()
            if image.max() > 0:  # Avoid division by zero
                image = image / image.max()
            image = (image * 255).astype(np.uint8)
            
        return image
    except Exception as e:
        st.error(f"Error converting image: {str(e)}")
        return None

# Main Streamlit app
def main():
    st.title("Comparative Assessment of Resolution Enhancement Models for Satellite Images")
    
    # Added info about the methods at the top
    with st.expander("About the Enhancement Methods", expanded=True):
        st.markdown("""
        ### Available Enhancement Methods:
        
        #### Deep Learning:
        - **CNN-SR**: Neural network-based super-resolution using a pre-trained model
        - **Wavelet-SR**: Combined wavelet sharpening with super-resolution
        - **Wiener-SR**: Combined Wiener filter with super-resolution
        
        #### Traditional Methods:
        - **Bicubic**: Standard bicubic interpolation for upscaling
        - **Lanczos**: Lanczos resampling for high-quality upscaling
        
        #### Sharpening Methods:
        - **Wavelet**: Wavelet-based image sharpening
        - **Wiener**: Wiener filter for image restoration
        """)
        
    # Add info about the pre-trained model at the top
    with st.expander("About the Pre-trained SR Model", expanded=True):
        st.markdown("""
        This application uses a pre-trained super-resolution model (`sr_model_x2.keras`) for all 
        super-resolution related methods. This model has been specifically trained to upscale images 
        by a factor of 2x while preserving and enhancing details.
        
        The model architecture uses a deep convolutional neural network with residual connections 
        to maintain image fidelity while adding details that might have been lost in the 
        lower-resolution version.
        
        For best results, use one of the SR-based methods:
        - **CNN-SR**: Uses the pre-trained model directly
        - **Wavelet-SR**: Combines SR with wavelet sharpening for enhanced details
        - **Wiener-SR**: Combines SR with Wiener filtering for noise reduction
        """)
    
    st.sidebar.header("Settings")
    
    # Verify dataset structure and update path if needed
    if not verify_dataset_structure():
        st.error("Please check your dataset folder structure and try again.")
        return
    
    # Get available categories
    available_categories = get_available_categories()
    if not available_categories:
        st.error("No categories found in the dataset!")
        return
    
    # Setup sidebar options
    st.sidebar.subheader("Image Selection")
    selected_category = st.sidebar.selectbox("Select Category", available_categories)
    
    # Get images from selected category
    category_path = os.path.join(DATASET_PATH, selected_category)
    
    if not os.path.exists(category_path):
        st.error(f"Category path not found: {category_path}")
        return
        
    image_files = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
    
    if not image_files:
        st.error(f"No images found in category: {selected_category}")
        return
    
    selected_image = st.sidebar.selectbox("Select Image", image_files)
    image_path = os.path.join(category_path, selected_image)
    
    # Select enhancement method - reordered to put SR methods first
    st.sidebar.subheader("Enhancement Method")
    method_names = [
        "CNN-SR", "Wavelet-SR", "Wiener-SR",  # Super-resolution methods first
        "Bicubic", "Lanczos",                 # Traditional methods
        "Wavelet", "Wiener"                   # Sharpening methods
    ]
    selected_method = st.sidebar.selectbox("Select Method", method_names)
    
    # Load image
    original_img, hr_img, lr_img = load_single_image(image_path)
    
    if original_img is None:
        st.error("Failed to load the selected image")
        return
    
    # Initialize or load model if needed
    sr_model = None
    # New line: use your trained model file
    trained_model_path = "sr_model_x2.keras"

    # Check if any method requires the SR model
    if any(sr_method in selected_method for sr_method in ["CNN-SR", "Wavelet-SR", "Wiener-SR"]):
        with st.spinner("Loading Super-Resolution model..."):
            try:
                # Try to load the pre-trained model first
                if os.path.exists(trained_model_path):
                    st.info(f"Using pre-trained model: {trained_model_path}")
                    sr_model = tf.keras.models.load_model(trained_model_path)
                # Fall back to other options if trained model doesn't exist
                elif os.path.exists("sr_model.h5"):
                    sr_model = tf.keras.models.load_model("sr_model.h5")
                else:
                    st.warning("No pre-trained model found. Building a new untrained model instead.")
                    sr_model = SuperResolutionModel.build_generator(scale_factor=SCALE_FACTOR)
                    sr_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), loss="mse")
                    st.info("Note: For best results, a trained model is recommended.")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.warning("Building a new model instead...")
                sr_model = SuperResolutionModel.build_generator(scale_factor=SCALE_FACTOR)
                sr_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), loss="mse")
    
    # Process button
    if st.sidebar.button("Process Image"):
        with st.spinner(f"Processing with {selected_method}..."):
            start_time = time.time()
            enhanced_img = process_image(lr_img, selected_method, sr_model)
            processing_time = time.time() - start_time
            
            if enhanced_img is not None:
                # Display images only (no metrics)
                st.header("Results")
                
                # Create three columns for the images
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Low Resolution Input")
                    resized_lr = cv2.resize(lr_img, (HR_SIZE, HR_SIZE))
                    st.image(convert_to_displayable(resized_lr), use_container_width=True)
                
                with col2:
                    st.subheader("High Resolution Ground Truth")
                    st.image(convert_to_displayable(hr_img), use_container_width=True)
                
                with col3:
                    st.subheader(f"Enhanced with {selected_method}")
                    st.image(convert_to_displayable(enhanced_img), use_container_width=True)
                
                # Show processing time only
                st.info(f"Processing Time: {processing_time:.3f} seconds")
                
                # Show image difference map
                st.subheader("Difference Map")
                diff_map = np.abs(hr_img - enhanced_img) * 3  # This multiplication by 3 might cause values > 1.0
                diff_map = np.clip(diff_map, 0, 1)  # Ensure values are in [0,1] range

                # Check for invalid values again
                if np.isnan(diff_map).any() or np.isinf(diff_map).any():
                    st.warning("Difference map contains invalid values. Fixing...")
                    diff_map = np.nan_to_num(diff_map, nan=0.0, posinf=1.0, neginf=0.0)

                st.image(convert_to_displayable(diff_map), use_container_width=True)
                
                # Option to download enhanced image
                buffered = io.BytesIO()
                enhanced_pil = Image.fromarray(convert_to_displayable(enhanced_img))
                enhanced_pil.save(buffered, format="PNG")
                st.download_button(
                    label="Download Enhanced Image",
                    data=buffered.getvalue(),
                    file_name=f"enhanced_{selected_method}_{selected_image}",
                    mime="image/png"
                )
    else:
        # Display original images before processing
        st.header("Original Images")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(original_img, use_container_width=True)
        
        with col2:
            st.subheader("Low Resolution Version")
            resized_lr = cv2.resize(lr_img, (HR_SIZE, HR_SIZE))
            st.image(convert_to_displayable(resized_lr), use_container_width=True)
            
        st.info("👆 Select an enhancement method and click 'Process Image' to see the enhanced result.")

if __name__ == "__main__":
    main()
