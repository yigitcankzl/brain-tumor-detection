from data_preparation import prepare_data
from train_model import train_model
from use_model import evaluate_model

def main():
    train_dir = "data/Training"
    test_dir = "data/Testing"
    
    print("Preparing data...")
    train_generator, test_generator = prepare_data(train_dir, test_dir)
    
    train_model_flag = input("Do you want to train the model? (Yes/No): ").strip().lower()
    
    if train_model_flag == "yes":
        print("Training the model...")
        train_model(train_generator, test_generator)
    else:
        print("Skipping model training, using the pre-trained model.")
    
    print("Evaluating the model...")
    evaluate_model("model/brain_tumor_detection_model.h5", test_generator)

if __name__ == "__main__":
    main() 