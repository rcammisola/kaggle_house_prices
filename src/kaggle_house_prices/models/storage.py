import pickle


def save_model(model, model_name):
    model_path = f"models/{model_name}/model.pickle"
    with open(model_path, "wb") as model_file_pointer:
        pickle.dump(model, model_file_pointer)