import numpy as np

class Cluster:
    def __init__(self, representative_vector) -> None:
        self.representative_vector:np.ndarray = representative_vector
        self.elements:np.ndarray = [[]]
        self.elements_index = []
        self.isLearningComplete = False

    def setElement(self, index:int, element:np.ndarray):
        self.elements_index.append(index)
        _element:np.ndarray = [element]
        if 0 < np.size(self.elements):
            self.elements = np.append(self.elements, _element, axis = 0)
        else:
            self.elements = _element
    
    def recalRepresentativeVector(self):
        self.isLearningComplete = np.all(self.representative_vector == np.mean(self.elements, axis = 0))
        self.representative_vector = np.mean(self.elements, axis = 0)
    
    def euclideanDistance(self, row:np.ndarray):
        return np.linalg.norm(self.representative_vector - row)
    
    def initElements(self):
        self.elements_index = []
        self.elements:np.ndarray = [[]]