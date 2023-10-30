from torchvision.models import resnet50, ResNet50_Weights
from torchvision.io import read_image


class resnetModel:
    def __init__(self):
        # Initialize model with the best available weights
        self.weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=self.weights)
        self.model.eval()

        # Initialize the inference transforms
        self.preprocess = self.weights.transforms()


    def vectorize(self, imgPath):
        # Apply inference preprocessing transforms
        img = read_image(imgPath)
        batch = self.preprocess(img).unsqueeze(0)

        # Use the model and print the predicted category
        prediction = self.model(batch).squeeze(0).softmax(0)
        predictionToArray = prediction.detach().numpy().tolist()
        return predictionToArray
    

    def categrize(self, imgPath):
        # Apply inference preprocessing transforms
        img = read_image(imgPath)
        batch = self.preprocess(img).unsqueeze(0)

        # Step 4: Use the model and print the predicted category
        prediction = self.model(batch).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        category_name = self.weights.meta["categories"][class_id]
        return [category_name, score]

# res = resnetModel()
# result = res.vectorize("D:/NJA/2023_ComputerEngineering_Project1/segment-anything/mySrc/3.jpg")
# print("length:" +  str(len(result)))
# print("type:" +  str(type(result[0])))