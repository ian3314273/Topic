import cv2 as cv
import os
import time
import numpy as np
import tensorflow as tf
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
import librosa
import soundfile as sf
import time
from ultralytics import YOLO
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from tensorflow.keras.models import load_model


source_path = "C:/Coding/Topic_One/Pose/models/research/audioset/vggish/20240518_144209 (1).mp4"
cap = cv.VideoCapture(source_path)

# Use YOLOv8 to detect people
person_model = YOLO('yolov8n.pt')
# Violence detection model
violence_model = YOLO('0527violence.pt')

frame_rate = 5  # Every 5 frames
frame_count = 0
using_violence_model = False

# For FPS calculation
fps_start_time = time.time()
fps_frame_count = 0

# Load video for audio extraction
video = VideoFileClip(source_path)
audio = video.audio
fps = cap.get(cv.CAP_PROP_FPS)
total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

# Directory to save audio clips
audio_dir = "audio_clips/"
output_dir = "output/"
os.makedirs(audio_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
def extract_vggish_features(audio_file, model_checkpoint, pca_params):
    # Load the audio file and process it to the format required by VGGish
    examples_batch = vggish_input.wavfile_to_examples(audio_file)

    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
        # Define the VGGish model in inference mode
        vggish_slim.define_vggish_slim(training=False)

        # Load the checkpoint
        vggish_slim.load_vggish_slim_checkpoint(sess, model_checkpoint)

        # Locate the input and output tensors
        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)

        # Start timing the feature extraction
        start_time = time.time()

        # Run inference
        [embedding_batch] = sess.run([embedding_tensor], feed_dict={features_tensor: examples_batch})

        # Postprocess the embeddings
        pproc = vggish_postprocess.Postprocessor(pca_params)
        postprocessed_batch = pproc.postprocess(embedding_batch)

        # End timing the feature extraction
        feature_extraction_time = time.time() - start_time
        print(f"Feature extraction time for {audio_file}: {feature_extraction_time:.4f} seconds")

    return postprocessed_batch, feature_extraction_time
def process_VGG(filename, output_dir, model_checkpoint, pca_params):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    total_extraction_time = 0
    num_files = 0
    if filename.endswith('.wav'):
        input_file = filename
        output_file = os.path.join(output_dir, f"{os.path.splitext(input_file)[0]}.npy")
        try:
            features, extraction_time = extract_vggish_features(input_file, model_checkpoint, pca_params)
            np.save(output_file, features)
            print(features)
            print(features.shape)
            print(f"Processed {input_file} and saved features to {output_file}")
            total_extraction_time += extraction_time
            num_files += 1
        except Exception as e:
            print(f"Error processing {input_file}: {e}")

    if num_files > 0:
        print(f"Average feature extraction time: {total_extraction_time / num_files:.4f} seconds per file")
def main():
    global frame_count, fps_frame_count, fps_start_time, using_violence_model
    while True:
        ret, frame = cap.read()
        if not ret:
            print("NO")
            break
        frame_count += 1
        fps_frame_count += 1
        if frame_count > total_frames:
            frame_count = 1

        if frame_count % frame_rate == 0:
            if using_violence_model:
                # Use violence detection model and filter for 'Violence' class (class 1)
                results = violence_model.predict(frame, classes=[1], show=True, verbose=False)  # class 1 is 'Violence'

                # Check if any 'Violence' (class 1) detected
                violence_detected = len(results[0].boxes) > 0
                if violence_detected:
                    # Extract and save audio for the previous second
                    current_time = frame_count / fps  # Current time in seconds
                    start_time = max(current_time - 1, 0)
                    end_time = current_time
                    audio_segment = audio.subclip(start_time, end_time)

                    # Save as temporary mp3 file
                    temp_mp3_path = f"{audio_dir}/temp_{int(current_time)}.mp3"
                    audio_segment.write_audiofile(temp_mp3_path, codec='mp3')

                    # Convert mp3 to mono wav using pydub
                    sound = AudioSegment.from_mp3(temp_mp3_path)
                    sound = sound.set_channels(1)  # Convert to mono
                    sound.export(f"audio_{int(current_time)}.wav", format="wav")
                    # Remove temporary mp3 file
                    os.remove(temp_mp3_path)
                    process_VGG(f"audio_{int(current_time)}.wav", output_dir,'vggish_model.ckpt', 'vggish_pca_params.npz')
                    sound_model = load_model('violence_detection_model.h5')
                    feature = np.load(f"output/audio_{int(current_time)}.npy")
                    sound_prediction = sound_model.predict(feature)
                    predicted_label = (sound_prediction > 0.5).astype(int)
                    if(predicted_label == 1):
                        print("Violence detected in audio")
                    else:
                        print("No violence detected in audio")
                if not violence_detected:
                    using_violence_model = False  # Switch back to YOLOv8 if no violence detected
            else:
                # Use YOLOv8 to detect people
                person_results = person_model.predict(frame, classes=[0], show=True, verbose=False)  # class 0 is 'person'
                person_count = len(person_results[0].boxes)

                if person_count >= 2:
                    using_violence_model = True  # Switch to violence detection model

        # Calculate and print FPS every second
        if time.time() - fps_start_time >= 1.0:
            print(f"FPS: {fps_frame_count}")
            fps_frame_count = 0
            fps_start_time = time.time()
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
