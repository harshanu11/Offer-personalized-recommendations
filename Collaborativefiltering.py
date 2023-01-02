from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split

# Load the dataset
data = Dataset.load_builtin('ml-100k')

# Split the dataset into training and testing sets
trainset, testset = train_test_split(data, test_size=0.25)

# Use the SVD algorithm to train the model
algo = SVD()
algo.fit(trainset)

# Make predictions on the test set
predictions = algo.test(testset)

# Calculate and print the accuracy of the predictions
accuracy.rmse(predictions)