import joblib
from sklearn.linear_model import LogisticRegression

b = joblib.load('outputs/1_06032026_111827_46782/model/06032026_111827_46782.pkl')

def search_dict(d, path=""):
    if isinstance(d, dict):
        for k, v in d.items():
            search_dict(v, path + f"['{k}']")
    elif isinstance(d, list):
        for i, v in enumerate(d):
            search_dict(v, path + f"[{i}]")
    elif isinstance(d, LogisticRegression):
        print(f"Found LogisticRegression at {path}")
    elif hasattr(d, '__dict__'):
        search_dict(d.__dict__, path + f".__dict__")

search_dict(b, "bundle")
