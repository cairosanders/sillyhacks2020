import pickle
import sys


def main():
    input_caption = ["i am an example caption"]

    vectorizer = pickle.load(open("vectorizer.pickle", "rb"))
    loaded_model = pickle.load(open("classifier.pickle", "rb"))
    label_encoder = pickle.load(open('label_encoder.pickle', 'rb'))
    input_caption = vectorizer.transform(input_caption)
    meme_code = loaded_model.predict(input_caption)
    meme_name = label_encoder.inverse_transform([int(meme_code)])
    print("predicted: ", meme_name)


if __name__ == "__main__":
    main()
    sys.exit()
