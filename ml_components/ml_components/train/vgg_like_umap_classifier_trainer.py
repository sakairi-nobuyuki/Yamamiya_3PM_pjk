# coding: utf-8

from typing import List, Dict, Any
import numpy as np
from scipy.spatial import ConvexHull
import os


from ..components.inference import UmapReducingPredictor, VggLikeFeatureExtractor
from ..components.dataloader import BinaryClassifierDataloaderFactory
from ..io import DataTransferS3, IOTemplate
from . import TemplateTrainer
from ..models.factory import ModelFactoryTemplate


class VggLikeUmapClassifierTrainer(TemplateTrainer):
    def __init__(self, data_path: str, factory: ModelFactoryTemplate, image_io: DataTransferS3, model_path: str = None, n_layer: int = -1) -> None:
        if not isinstance(factory, ModelFactoryTemplate):
            raise TypeError(f"{type(factory) is not {ModelFactoryTemplate}}")

        print("VggLikeUmapPredictor")
        self.image_io = image_io
        self.factory = factory
        # TODO: create data path list and label list


        # Train the model on your dataset using binary cross-entropy loss and SGD optimizer
        print(">> model path: ", model_path)
        if model_path is None:
            # TODO: create a temporary vgg model at a model_path
            pass
        print(">> n layer: ", n_layer)
        #self.vgg = VggLikeFeatureExtractor(model_path, self.factory, n_layer)
        #self.reducer = UmapReducingPredictor()


    def configure_dataset(self, data_path: str, target: str) -> Dict[str, List[str]]:
        """Create a dict of lists in data_path in the bucket such that
        {
            "label_1": [file_11, file_12, ...],
            "label_2": [file_21, file_22, ...],
            ...
        }

        Args:
            target (str): "train", "val", or "test"

        Returns:
            Dict[str, List[str]]: _description_
        """
        
        data_path_list = self.image_io.get_blob()
        # print("data path list as is: ", data_path_list)
        # extract train
        data_path_list = [data_path for data_path in data_path_list if os.path.join(data_path, target) in data_path_list]
        print("data path list filtered: ", data_path_list)

        # sort data according to labels
        # get label list
        #data_path_list = list(set([data_path.split("/")[-1] for data_path in data_path_list]))

        data_path_list_dict = {data_path.split("/")[-1]: data_path for data_path in data_path_list}

        return data_path_list_dict


    #def train(self, image_list_dict: Dict[str, List[np.ndarray]]) -> Dict[str, Any]:
    def train(self) -> Dict[str, Any]:
        """Train the UMAP parameter so that it will minimize the distance between clusters.
        - Create a list of combinations of two classes.
        - Calculate distances of each combinations.
        - Pertub the parameters and optimize so that it will minimize the total distances.

        Args:
            image_list_dict (Dict[List[np.ndarray]]): Input images list of classes

        Returns:
            Dict[str, Any]: UMAP parameters
        """
        for inputs, labels in self.dataloader.train_loader:
            
            for i in range(len(inputs)):
                inp = inputs[i].numpy()
                print(type(inp), inp.shape)
                feature = self.vgg.predict(inp)
            #feature_list = [self.vgg.predict(inputs[i].numpy()) for i in range(len(inputs))]
            labels_list = list(labels.numpy())
            print(labels_list)
#        with tqdm(dataloader, total=len(dataloader)) as train_progress:
#            for inputs, labels in train_progress:
#
#                inputs = inputs.to(self.device)
#                labels = labels.to(self.device)
#                self.optimizer.zero_grad()
#                outputs = model(inputs)
#                loss = self.criterion(outputs, labels)
#                loss.backward()
#                self.optimizer.step()
#                train_loss += loss.item()
#        train_loss = train_loss / float(len(dataloader))
#        return train_loss

        pass

#    def calculate_cluster_haussdorf_distance(self, input_1: np.ndarray, input_2: np.ndarray) -> float:
#        """Calculate Haussdorf distance of two np.arrays, which was clusterized with UMAP.
#
#        Args:
#            input_1 (np.ndarray): UMAP reduced features
#            input_2 (np.ndarray): UMAP reduced features
#
#        Returns:
#            float: Distance between input_1 and input_2 with Haussdorf distance
#        """

#        print(input_1[0], input_1[1], input_1)

#    def get_convex_hull(self, vertices: np.ndarray) -> np.ndarray:
#        return ConvexHull(vertices)

#    def calculate_d0_cluster_simplices(self, vertices_1: np.ndarray, vertices_2: np.ndarray) -> int:
#
#        hull_1 = self.get_convex_hull(vertices_1)
#        hull_2 = self.get_convex_hull(vertices_2)
#
#        print(hull_1.simplices)
#        print(hull_2.simplices)
#        for simplex_1 in hull_1.simplices:
#            print("simplex_1: ", simplex_1.ve)
#        for simplex_2 in hull_2.simplices:
#            print(simplex_1, simplex_2)
        #        if set(simplex_1).intersection(set(simplex_2)):
        #            return 0
        #return 1


    def fit_all(self, image_list_dict: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
        return {
            image_list_key: self.fit(image_list)
            for image_list_key, image_list in image_list_dict.items()
        }

    def fit(self, image_list: List[np.ndarray]) -> np.ndarray:
        """Fit the images of a cluster that is supervised to classes by human, and
        fit by UMAP

        Args:
            image_list (List[np.ndarray]): A list of images of OpenCV

        Returns:
            np.ndarray: Fit and transformed result by UMAP
        """

        feat_list = [self.vgg.predict(image) for image in image_list]
        feat_array = np.concatenate(feat_list)

        reduced_feat = self.reducer.predict(feat_array)

        return reduced_feat
