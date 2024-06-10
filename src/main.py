from data_processing import preprocess_category_data, preprocess_sentiment_data
from model_training import MultiLabelDataset, SentimentDataset, train_category_model, train_sentiment_model
from prediction import predict_sentiment

def main():
    # preprocess data
    X_train_cat, X_eval_cat, y_train_cat, y_eval_cat = preprocess_category_data('./category.csv')
    X_train_sen, X_eval_sen, y_train_sen, y_eval_sen = preprocess_sentiment_data('./sentiment.csv')

    # train category model
    train_dataset_cat = MultiLabelDataset(X_train_cat, y_train_cat)
    eval_dataset_cat = MultiLabelDataset(X_eval_cat, y_eval_cat)
    category_model = train_category_model(train_dataset_cat, eval_dataset_cat)

    # train sentiment model
    train_dataset_sen = SentimentDataset(X_train_sen, y_train_sen)
    eval_dataset_sen = SentimentDataset(X_eval_sen, y_eval_sen)
    sentiment_model = train_sentiment_model(train_dataset_sen, eval_dataset_sen)

    # input sentences
    input_strings = [["Good price, great service"],["Was messed around quite a lot - times rearranged due to tyres not being delivered, then put back to the original appointment - without cancellung the changed one. Will go directly through the garage next time. [REDACTED] messed me about too much for the same price as the garage would have given without them."],["Excellent service and saved a few pounds Will definitely be using [REDACTED] again."],["Competitive prices and consistently excellent. This has been the case for many years. No problems with the tyres being sent on time to the fitters or the fitters themselves."]]

    # make predictions
    for input_string in input_strings:
        print(input_string)
        predicted_sentiments = predict_sentiment(input_string, category_model, sentiment_model)
        print(predicted_sentiments)
        print("\n")

if __name__ == "__main__":
    main()