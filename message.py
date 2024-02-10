import cv2
import numpy as np
import tensorflow as tf
import os

# Function to average circles detected
def avg_circles(circles, b):
    avg_x = avg_y = avg_r = 0
    for i in range(b):
        avg_x += circles[0][i][0]
        avg_y += circles[0][i][1]
        avg_r += circles[0][i][2]
    return int(avg_x/b), int(avg_y/b), int(avg_r/b)

# Function to calculate distance between two points
def dist_2_pts(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Function to execute k-means clustering
def k_means(coordinates, k, sample_color, max_iterations=100):
    centroids = coordinates[np.random.choice(len(coordinates), k, replace=False)]
    for _ in range(max_iterations):
        distances = np.linalg.norm(coordinates[:, np.newaxis] - centroids, axis=2)
        assignments = np.argmin(distances, axis=1)
        new_centroids = np.array([np.mean(coordinates[assignments == i], axis=0) for i in range(k)])
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    RGB_cluster = np.argmin(np.linalg.norm(centroids - np.array(sample_color), axis=1))
    boolean_assignments = assignments == RGB_cluster
    return boolean_assignments, assignments

# Function to get circle border pixels
def get_circle_border_pixels(image, center_x, center_y, radius):
    height, width, _ = image.shape
    y_coords, x_coords = np.ogrid[:height, :width]
    distance_map = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    mask = np.abs(distance_map - radius) < 1.5
    border_pixels = np.argwhere(mask)
    border_pixel_values = image[border_pixels[:, 0], border_pixels[:, 1]]
    return border_pixels, border_pixel_values

def preprocess_image(image):
    # Resize the image to match the model's expected input size
    processed_image = cv2.resize(image, (input_details[0]['shape'][2], input_details[0]['shape'][1]))

    # Ensure the image is in the correct format (UINT8)
    processed_image = processed_image.astype(np.uint8)

    # Expand dimensions to match the model's input shape
    processed_image = np.expand_dims(processed_image, axis=0)

    return processed_image

def predict(image):
    processed_image = preprocess_image(image)

    interpreter.set_tensor(input_details[0]['index'], processed_image)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


# Load the model
interpreter = tf.lite.Interpreter(model_path="Analog_Gauge_Images_Reader-main\gate-classificationV3.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(1)

while True:
    try:
        ret, img = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        if img is not None:
            cv2.imshow("Processed Image", img)
            try:
                
                # Make a prediction using the TFLite model
                prediction = predict(img)
                label_index = np.argmax(prediction)
                labels = ['white', 'black', 'silver']
                predicted_label = labels[label_index]

                if predicted_label == 'black':  # Define condition_1
                    min_angle, max_angle = 46, 315
                    min_value, max_value = 50, 800
                    height_multiply_1, height_multiply_2 = 0.3, 0.45
                    units = 'F'
                elif predicted_label == 'silver':  # Define condition_2
                    min_angle, max_angle = 45, 313
                    min_value, max_value = 100, 1000
                    height_multiply_1, height_multiply_2 = 0.3, 0.4
                    units = 'F'
                elif predicted_label == 'white':  # Default condition
                    min_angle, max_angle = 30, 330
                    min_value, max_value = -10, 110
                    height_multiply_1, height_multiply_2 = 0.25, 0.30
                    units = 'C'
                else:
                    print('False condition')
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                height, width = img.shape[:2]
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, np.array([]), 100, 50, int(height*height_multiply_1), int(height*height_multiply_2))
                if circles is not None and len(circles)>0:
                    a, b, c = circles.shape
                    x, y, r = avg_circles(circles, b)
                else:
                    continue
                
                # Final calculations
                gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                red_channel = img[:,:,2]
                red_thresh = 70
                dst2 = red_channel < red_thresh
                dst2 = (dst2 * 255).astype('uint8')

                measurement_r = int(r * 1.6 / 5)
                measurement_x = x 
                measurement_y = y 
                border_pixels_coordinate, border_pixel_values = get_circle_border_pixels(img, measurement_x, measurement_y, measurement_r)
                k_means_bool_result, _ = k_means(border_pixel_values, k=5, sample_color=[32, 31, 130], max_iterations=10000)

                avg_x = avg_y = count = 0
                for i, is_needle in enumerate(k_means_bool_result):
                    if is_needle:
                        y, x = border_pixels_coordinate[i]
                        avg_x += x
                        avg_y += y
                        count += 1
                        cv2.circle(img, (x, y), 1, (255, 0, 0), 5, cv2.LINE_AA)

                if count > 0:
                    avg_x /= count
                    avg_y /= count
                else:
                    continue

                y_angle = measurement_y - avg_y
                x_angle = avg_x - measurement_x
                res = np.arctan2(y_angle, x_angle)
                res = np.rad2deg(res)

                if x_angle > 0 and y_angle > 0:
                    final_angle = 270 - res
                elif x_angle < 0 and y_angle > 0:
                    final_angle = 90 - res
                elif x_angle < 0 and y_angle < 0:
                    final_angle = 90 - res
                elif x_angle > 0 and y_angle < 0:
                    final_angle = 270 - res

                old_range = (max_angle - min_angle)
                new_range = (max_value - min_value)
                new_value = (((final_angle - min_angle) * new_range) / old_range) + min_value

                cv2.putText(img, f'Value: {predicted_label} {new_value} {units}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow("Processed Image", img) #with cv2_imshow(img)
                # cv2.waitKey(0) # Disable this line to have continuous detechtion

            except Exception as e:
                print(f"Error processing {cap}: {e}")
    except Exception as e:
        print("Error:", e)
    finally:
        if cv2.waitKey(1) == ord('q'):
            break
cv2.destroyAllWindows()