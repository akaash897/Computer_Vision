import streamlit as st
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, LeakyReLU, BatchNormalization, Input
from tensorflow.keras.models import Model
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

        if len(img_float.shape) == 3:
            r, g, b = cv2.split(img_float)
            r_coeffs = pywt.dwt2(r, 'haar')
            g_coeffs = pywt.dwt2(g, 'haar')
            b_coeffs = pywt.dwt2(b, 'haar')
            r_sharp = pywt.idwt2((r_coeffs[0], (r_coeffs[1][0]*1.5, r_coeffs[1][1]*1.5, r_coeffs[1][2]*1.5)), 'haar')
            g_sharp = pywt.idwt2((g_coeffs[0], (g_coeffs[1][0]*1.5, g_coeffs[1][1]*1.5, g_coeffs[1][2]*1.5)), 'haar')
            b_sharp = pywt.idwt2((b_coeffs[0], (b_coeffs[1][0]*1.5, b_coeffs[1][1]*1.5, b_coeffs[1][2]*1.5)), 'haar')
            sharpened = cv2.merge([r_sharp, g_sharp, b_sharp])
        else:
            coeffs = pywt.dwt2(img_float, 'haar')
            sharpened = pywt.idwt2((coeffs[0], (coeffs[1][0]*1.5, coeffs[1][1]*1.5, coeffs[1][2]*1.5)), 'haar')

        return np.clip(sharpened, 0, 255).astype(np.uint8)

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
    def laplacian_sharpening(image, alpha=0.5):
        """Apply Laplacian-based sharpening"""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            laplacian = cv2.Laplacian(l, cv2.CV_64F)
            sharpened_l = np.clip(l - alpha * laplacian, 0, 255).astype(np.uint8)
            enhanced_lab = cv2.merge([sharpened_l, a, b])
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        else:
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            return np.clip(image - alpha * laplacian, 0, 255).astype(np.uint8)

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

    @staticmethod
    def laplacian_sr_hybrid(sr_image):
        """Combine Super-Resolution with Laplacian Sharpening"""
        img_uint8 = (sr_image * 255).astype(np.uint8)
        sharpened = EnhancementMethods.laplacian_sharpening(img_uint8)
        return sharpened.astype(np.float32) / 255.0

class EvaluationMetrics:
    """Methods to evaluate image enhancement quality"""

    @staticmethod
    def calculate_psnr(img1, img2):
        """Calculate Peak Signal-to-Noise Ratio"""
        return psnr(img1, img2)

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
        if method_name == "Bicubic":
            return InterpolationMethods.bicubic(lr_img, scale=SCALE_FACTOR)
        elif method_name == "Lanczos":
            return InterpolationMethods.lanczos(lr_img, scale=SCALE_FACTOR)
        elif method_name == "Super-Resolution":
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
            return np.clip(sr_img, 0, 1)
        elif method_name == "Wavelet":
            upscaled = cv2.resize(lr_img, (HR_SIZE, HR_SIZE))
            img_uint8 = (upscaled * 255).astype(np.uint8)
            enhanced = EnhancementMethods.wavelet_sharpening(img_uint8)
            return enhanced.astype(np.float32) / 255.0
        elif method_name == "Wiener":
            upscaled = cv2.resize(lr_img, (HR_SIZE, HR_SIZE))
            img_uint8 = (upscaled * 255).astype(np.uint8)
            enhanced = EnhancementMethods.wiener_filter(img_uint8)
            return enhanced.astype(np.float32) / 255.0
        elif method_name == "Laplacian":
            upscaled = cv2.resize(lr_img, (HR_SIZE, HR_SIZE))
            img_uint8 = (upscaled * 255).astype(np.uint8)
            enhanced = EnhancementMethods.laplacian_sharpening(img_uint8)
            return enhanced.astype(np.float32) / 255.0
        elif method_name == "Wavelet-SR":
            if model is None:
                st.error("Super-Resolution model not loaded")
                return None
            
            # First get the SR result
            lr_batch = np.expand_dims(lr_img, axis=0)
            sr_img = model.predict(lr_batch)[0]
            sr_img = np.clip(sr_img, 0, 1)
            # Then apply wavelet sharpening
            return EnhancementMethods.wavelet_sr_hybrid(sr_img)
        elif method_name == "Wiener-SR":
            if model is None:
                st.error("Super-Resolution model not loaded")
                return None
            
            # First get the SR result
            lr_batch = np.expand_dims(lr_img, axis=0)
            sr_img = model.predict(lr_batch)[0]
            sr_img = np.clip(sr_img, 0, 1)
            # Then apply wiener filter
            return EnhancementMethods.wiener_sr_hybrid(sr_img)
        elif method_name == "Laplacian-SR":
            if model is None:
                st.error("Super-Resolution model not loaded")
                return None
            
            # First get the SR result
            lr_batch = np.expand_dims(lr_img, axis=0)
            sr_img = model.predict(lr_batch)[0]
            sr_img = np.clip(sr_img, 0, 1)
            # Then apply laplacian sharpening
            return EnhancementMethods.laplacian_sr_hybrid(sr_img)
        else:
            st.error(f"Unknown method: {method_name}")
            return None
    
    except Exception as e:
        st.error(f"Error processing image with {method_name}: {str(e)}")
        return None

def calculate_metrics(enhanced_img, hr_img):
    """Calculate image quality metrics"""
    try:
        ssim_val = EvaluationMetrics.calculate_ssim(enhanced_img, hr_img)
        psnr_val = EvaluationMetrics.calculate_psnr(enhanced_img, hr_img)
        mse_val = EvaluationMetrics.calculate_mse(enhanced_img, hr_img)
        return ssim_val, psnr_val, mse_val
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return 0, 0, 0

def convert_to_displayable(image):
    """Convert image to format suitable for display"""
    try:
        if image is None:
            return None
            
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        return image
    except Exception as e:
        st.error(f"Error converting image: {str(e)}")
        return None

# Main Streamlit app
def main():
    st.title("Image Enhancement and Super-Resolution App")
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
    
    # Select enhancement method
    st.sidebar.subheader("Enhancement Method")
    method_names = [
        "Bicubic", "Lanczos", "Super-Resolution",
        "Wavelet", "Wiener", "Laplacian",
        "Wavelet-SR", "Wiener-SR", "Laplacian-SR"
    ]
    selected_method = st.sidebar.selectbox("Select Method", method_names)
    
    # Load image
    original_img, hr_img, lr_img = load_single_image(image_path)
    
    if original_img is None:
        st.error("Failed to load the selected image")
        return
    
    # Initialize or load model if needed
    sr_model = None
    model_path = "sr_model.h5"
    if "Super-Resolution" in selected_method:
        if os.path.exists(model_path):
            with st.spinner("Loading Super-Resolution model..."):
                try:
                    sr_model = tf.keras.models.load_model(model_path)
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
                    st.warning("Building a new model instead...")
                    sr_model = SuperResolutionModel.build_generator(scale_factor=SCALE_FACTOR)
                    sr_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), loss="mse")
        else:
            with st.spinner("Building Super-Resolution model..."):
                sr_model = SuperResolutionModel.build_generator(scale_factor=SCALE_FACTOR)
                sr_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), loss="mse")
                st.info("Note: In a real application, you would train the model on your dataset. Here we're just using the untrained model for demonstration.")
    
    # Process button
    if st.sidebar.button("Process Image"):
        with st.spinner(f"Processing with {selected_method}..."):
            start_time = time.time()
            enhanced_img = process_image(lr_img, selected_method, sr_model)
            processing_time = time.time() - start_time
            
            if enhanced_img is not None:
                # Calculate metrics
                ssim_val, psnr_val, mse_val = calculate_metrics(enhanced_img, hr_img)
                
                # Display images and metrics
                st.header("Results")
                
                # Create three columns for the images
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Low Resolution Input")
                    resized_lr = cv2.resize(lr_img, (HR_SIZE, HR_SIZE))
                    st.image(convert_to_displayable(resized_lr), use_column_width=True)
                
                with col2:
                    st.subheader("High Resolution Ground Truth")
                    st.image(convert_to_displayable(hr_img), use_column_width=True)
                
                with col3:
                    st.subheader(f"Enhanced with {selected_method}")
                    st.image(convert_to_displayable(enhanced_img), use_column_width=True)
                
                # Display metrics
                st.subheader("Image Quality Metrics")
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                
                with metrics_col1:
                    st.metric("SSIM", f"{ssim_val:.4f}")
                
                with metrics_col2:
                    st.metric("PSNR (dB)", f"{psnr_val:.2f}")
                
                with metrics_col3:
                    st.metric("MSE", f"{mse_val:.6f}")
                
                with metrics_col4:
                    st.metric("Processing Time (s)", f"{processing_time:.3f}")
                
                # Show image difference map
                st.subheader("Difference Map")
                diff_map = np.abs(hr_img - enhanced_img) * 3
                diff_map = np.clip(diff_map, 0, 1)
                st.image(convert_to_displayable(diff_map), use_column_width=True)
                
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
            st.image(original_img, use_column_width=True)
        
        with col2:
            st.subheader("Low Resolution Version")
            resized_lr = cv2.resize(lr_img, (HR_SIZE, HR_SIZE))
            st.image(convert_to_displayable(resized_lr), use_column_width=True)
            
        st.info("👆 Select an enhancement method and click 'Process Image' to see the enhanced result.")

    # Add info about the methods
    with st.expander("About the Enhancement Methods"):
        st.markdown("""
        ### Available Enhancement Methods:
        
        #### Traditional Methods:
        - **Bicubic**: Standard bicubic interpolation for upscaling
        - **Lanczos**: Lanczos resampling for high-quality upscaling
        
        #### Deep Learning:
        - **Super-Resolution**: Neural network-based super-resolution
        
        #### Sharpening Methods:
        - **Wavelet**: Wavelet-based image sharpening
        - **Wiener**: Wiener filter for image restoration
        - **Laplacian**: Laplacian sharpening for edge enhancement
        
        #### Hybrid Methods:
        - **Wavelet-SR**: Combined wavelet sharpening with super-resolution
        - **Wiener-SR**: Combined Wiener filter with super-resolution
        - **Laplacian-SR**: Combined Laplacian sharpening with super-resolution
        """)

if __name__ == "__main__":
    main()