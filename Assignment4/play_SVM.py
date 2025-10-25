import argparse

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def get_arg():
    parser = argparse.ArgumentParser(description="Just for fun!")
    parser.add_argument("--kernel", type=str, default="linear", choices=["linear", "rbf", "poly", "sigmoid"])
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arg()

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel=args.kernel, C=args.C, gamma=args.gamma))
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = args.test_size, random_state=args.seed, stratify=y
    )
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    # print("ACC: ", acc)
    # # print(f"ACC: {acc:.4f}")
    print("ACC: {:.4f}".format(
        acc
    ))