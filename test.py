from catboost import CatBoostClassifier
import pandas as pd

import sys


def load_models(path_to_star, path_to_type):
 
    model_star_l = CatBoostClassifier()
    model_star_l.load_model(path_to_star, format='cbm')
 
    model_type_l = CatBoostClassifier()
    model_type_l.load_model(path_to_type, format='cbm')
                
    return model_type_l, model_star_l


def comment_analyzer(comment, path_to_star, path_to_type):
    def clean_symbols(text):
        return ''.join(c for c in text if c.isalpha() or c.isspace())

    comment = [clean_symbols(' '.join(comment))]
 
    model_type, model_star = load_models(path_to_star, path_to_type)
    data = pd.DataFrame(comment, columns=['review'])
 
    rating = model_star.predict(data)  # 1 - 10
    comment_type = model_type.predict(data)  # 0 - neg / 1 - pos
 
    print('comment: {0}'.format(comment[0]))
    print('star: {0}, type: {1}' .format(rating[0][0], comment_type[0]))
  

if len(sys.argv) < 4:
    print("Usage: python test.py <path_to_model_star> <path_to_model_type> <comment>")
    sys.exit(0)

comment_analyzer(sys.argv[3:], sys.argv[1], sys.argv[2])