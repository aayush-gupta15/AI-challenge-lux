import numpy as np
from numpy.linalg import norm
import pickle
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
from annoy import AnnoyIndex
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2 as cv
from scipy import spatial
import copy
import argparse




class AppleDetector:
    def __init__(self):
        self.model = ResNet50(weights="resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5", include_top=False,
                              input_shape=(224, 224, 3))
        self.__input_shape = (224, 224, 3)
        self.__n_nearest_neighbors = 2
        self.__metric = 'euclidean'
        self.__trees = 150
        self.__gt_data_path = r"apple_meta_data.pickle"
        self.__gt_data = pickle.load(open(self.__gt_data_path, 'rb'))
        self.__extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

    def __get_image_feature_vector(self, image):
        # Calculate the image feature vector of the img
        features = self.model.predict(image)

        flattened_features = features.flatten()
        normalized_features = flattened_features / norm(flattened_features)

        return normalized_features

    def __load_image(self, path):
        img = image.load_img(path, target_size=(self.__input_shape[0], self.__input_shape[1]))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)

        return preprocessed_img

    def __get_file_list(self, root_dir):
        file_list = []

        for root, directories, filenames in os.walk(root_dir):
            for filename in filenames:
                if any(ext in filename for ext in self.__extensions):
                    file_list.append(os.path.join(root, filename))

        return file_list

    def __get_image_feature_vector_using_image_path(self, image_path):
        # Loads and pre-process the image
        img = self.__load_image(image_path)

        return self.__get_image_feature_vector(img)

    def get_format(self, image_path, img_name):
        image = cv.imread(image_path)
        image = cv.resize(image, (self.__input_shape[0], self.__input_shape[1]))

        expanded_img_array = np.expand_dims(image, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)

        feature_list = copy.deepcopy(self.__gt_data["apple"]["Features"])
        
        file_names = copy.deepcopy(self.__gt_data["apple"]["FileNames"])
        labels = copy.deepcopy(self.__gt_data["apple"]["Labels"])
        master_vector = self.__get_image_feature_vector(preprocessed_img)
        t = AnnoyIndex(len(master_vector), metric=self.__metric)

        for i in range(len(feature_list)):
            t.add_item(i, feature_list[i])

        i = i + 1
        t.add_item(i, master_vector)
        _ = t.build(self.__trees)

        feature_list.append(master_vector)
        file_names.append("Test")
        labels.append("Test")
        df = pd.DataFrame({'img_id': file_names, 'img_repr': feature_list, 'label': labels})

        base_img_id, base_vector, base_label = df.iloc[i, [0, 1, 2]]
        similar_img_ids = t.get_nns_by_item(i, self.__n_nearest_neighbors)
        similar_img_ids = similar_img_ids[1:]

        predicted_table_vector, predicted_table_format = df.iloc[similar_img_ids[0], [1, 2]]
        similarity = 1 - spatial.distance.cosine(base_vector, predicted_table_vector)
        rounded_similarity = int((similarity * 10000)) / 10000.0
        if rounded_similarity > 0.5:
            print("*"*30)
            print(f"Image {img_name} is an Apple")
            print("*"*30)
        else:
            print("*"*30)
            print(f"Image {img_name} is Not an Apple")
            print("*"*30)
        return

    def generate_metadata(self, train_data_path):
        feature_data = {}
        feature_data["apple"] = {}
        root_dir = train_data_path + os.sep + "apple"
        filenames = sorted(self.__get_file_list(root_dir))
        print("Total files: ", len(filenames))
        train_datagen = ImageDataGenerator(rescale=1. / 255)
        feature_list = []
        file_names = []
        labels = []

        for files in filenames:
            file_names.append(os.path.basename(files))
            labels.append("apple")

        for i in range(len(filenames)):
            feature_list.append(self.__get_image_feature_vector_using_image_path(filenames[i]))

        feature_data["apple"]["Features"] = feature_list
        feature_data["apple"]["FileNames"] = file_names
        feature_data["apple"]["Labels"] = labels

        with open("apple_meta_data.pickle", 'wb') as output:
            pickle.dump(feature_data, output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A test program.')

    parser.add_argument("-t", "--path", help="Enter path of image", default="")

    args = parser.parse_args()

    img_path = args.path
    if img_path:
        file_name = os.path.basename(img_path)
        my_class = AppleDetector()
        # my_class.generate_metadata(r"train")
        img = open(img_path, 'rb').read()
        my_class.get_format(img, file_name)