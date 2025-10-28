"""
Model Converter Script
This script converts your old model to a compatible format
Run this ONCE to fix the compatibility issue
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import h5py
import json

# ==================== CONFIGURATION ====================
OLD_MODEL_PATH = "final_rul_model_fixed.h5"  # Your current model
NEW_MODEL_PATH = "final_rul_model_compatible.h5"  # New compatible model
WINDOW_SIZE = 60

print("=" * 60)
print("MODEL CONVERTER - TensorFlow Compatibility Fixer")
print("=" * 60)

# ==================== METHOD 1: Extract and Rebuild ====================
print("\n[METHOD 1] Attempting to extract architecture and weights...")

try:
    # Read model configuration
    with h5py.File(OLD_MODEL_PATH, 'r') as f:
        # Get model config
        if 'model_config' in f.attrs:
            config_str = f.attrs['model_config']
            if isinstance(config_str, bytes):
                config_str = config_str.decode('utf-8')
            config = json.loads(config_str)
            print("✅ Model configuration extracted successfully")
            
            # Get input shape from config
            try:
                first_layer = config['config']['layers'][0]
                if 'batch_input_shape' in first_layer['config']:
                    input_shape = tuple(first_layer['config']['batch_input_shape'][1:])
                elif 'batch_shape' in first_layer['config']:
                    input_shape = tuple(first_layer['config']['batch_shape'][1:])
                else:
                    input_shape = None
                
                if input_shape:
                    print(f"✅ Input shape detected: {input_shape}")
                    WINDOW_SIZE = input_shape[0]
                    NUM_FEATURES = input_shape[1]
                else:
                    print("⚠️  Using default shape from training code")
                    NUM_FEATURES = 13  # Default from your training
            except:
                print("⚠️  Could not extract input shape, using defaults")
                NUM_FEATURES = 13
        else:
            print("⚠️  No model config found, using architecture from training code")
            NUM_FEATURES = 13
    
    # Reconstruct model architecture (from your training code)
    print(f"\n📐 Rebuilding model with shape: ({WINDOW_SIZE}, {NUM_FEATURES})")
    
    new_model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(WINDOW_SIZE, NUM_FEATURES)),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    
    # Build the model
    new_model.build(input_shape=(None, WINDOW_SIZE, NUM_FEATURES))
    print("✅ Model architecture rebuilt")
    
    # Load weights from old model
    print("\n📦 Loading weights from old model...")
    try:
        new_model.load_weights(OLD_MODEL_PATH)
        print("✅ Weights loaded successfully")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        print("\nTrying alternative weight loading method...")
        
        # Try loading old model and copying weights
        try:
            old_model = load_model(OLD_MODEL_PATH, compile=False, safe_mode=False)
            new_model.set_weights(old_model.get_weights())
            print("✅ Weights copied from old model")
        except Exception as e2:
            print(f"❌ Alternative method also failed: {e2}")
            raise Exception("Could not load weights")
    
    # Compile the new model
    new_model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    print("✅ Model compiled")
    
    # Save the new model
    print(f"\n💾 Saving compatible model to: {NEW_MODEL_PATH}")
    new_model.save(NEW_MODEL_PATH)
    print("✅ New model saved successfully!")
    
    # Verify the new model
    print("\n🔍 Verifying new model...")
    test_model = load_model(NEW_MODEL_PATH, compile=False)
    test_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    print("✅ New model loads without errors!")
    
    print("\n" + "=" * 60)
    print("SUCCESS! ✅")
    print("=" * 60)
    print(f"\nYour new compatible model is saved at:")
    print(f"  📁 {NEW_MODEL_PATH}")
    print(f"\nUpdate your Streamlit app:")
    print(f'  MODEL_PATH = "{NEW_MODEL_PATH}"')
    print("\nModel Details:")
    print(f"  - Window Size: {WINDOW_SIZE}")
    print(f"  - Features: {NUM_FEATURES}")
    print(f"  - Architecture: 3 LSTM layers + 2 Dense layers")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ Method 1 failed: {e}")
    print("\n" + "=" * 60)
    print("Trying METHOD 2: Direct Conversion")
    print("=" * 60)
    
    # ==================== METHOD 2: Direct Save ====================
    try:
        print("\n[METHOD 2] Attempting direct model conversion...")
        
        # Load with various options
        old_model = load_model(OLD_MODEL_PATH, compile=False, safe_mode=False)
        print("✅ Old model loaded")
        
        # Recompile
        old_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        print("✅ Model compiled")
        
        # Save in new format
        old_model.save(NEW_MODEL_PATH, save_format='h5')
        print(f"✅ Saved to: {NEW_MODEL_PATH}")
        
        # Verify
        test_model = load_model(NEW_MODEL_PATH)
        print("✅ Verification successful!")
        
        print("\n" + "=" * 60)
        print("SUCCESS! ✅")
        print("=" * 60)
        print(f"\nUpdate your Streamlit app:")
        print(f'  MODEL_PATH = "{NEW_MODEL_PATH}"')
        print("=" * 60)
        
    except Exception as e2:
        print(f"\n❌ Method 2 also failed: {e2}")
        print("\n" + "=" * 60)
        print("SOLUTION REQUIRED")
        print("=" * 60)
        print("\n⚠️  Your model file has severe compatibility issues.")
        print("\nRecommended Actions:")
        print("\n1. BEST SOLUTION: Retrain your model")
        print("   - Run your training script again")
        print("   - It will generate a fresh, compatible model")
        print("   - Use the new model in your Streamlit app")
        print("\n2. Alternative: Check TensorFlow version")
        print("   - Training TF version: ???")
        print("   - Current TF version: " + tf.__version__)
        print("   - Try matching versions")
        print("\n3. Provide more info:")
        print("   - What TensorFlow version was used for training?")
        print("   - Can you share the original training script?")
        print("   - Do you have the model in .keras format?")
        print("\n" + "=" * 60)

# ==================== PRINT SYSTEM INFO ====================
print("\n" + "=" * 60)
print("SYSTEM INFORMATION")
print("=" * 60)
print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {tf.keras.__version__}")
print(f"Python Version: {tf.version.VERSION}")
print("=" * 60)