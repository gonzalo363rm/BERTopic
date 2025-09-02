# Monkey patch para compatibilidad con scikit-learn >= 1.0
# Este patch debe ejecutarse antes de importar cualquier módulo que use sklearn
try:
    from sklearn.feature_extraction.text import CountVectorizer
    # Si get_feature_names no existe, crear un alias para get_feature_names_out
    if not hasattr(CountVectorizer, 'get_feature_names'):
        CountVectorizer.get_feature_names = CountVectorizer.get_feature_names_out
except ImportError:
    pass

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    # Si get_feature_names no existe, crear un alias para get_feature_names_out
    if not hasattr(TfidfVectorizer, 'get_feature_names'):
        TfidfVectorizer.get_feature_names = TfidfVectorizer.get_feature_names_out
except ImportError:
    pass

# Hacer el patch de manera más robusta para todas las clases que hereden de _VectorizerMixin
try:
    from sklearn.feature_extraction.text import _VectorizerMixin
    # Solo hacer el patch si get_feature_names_out existe
    if hasattr(_VectorizerMixin, 'get_feature_names_out') and not hasattr(_VectorizerMixin, 'get_feature_names'):
        _VectorizerMixin.get_feature_names = _VectorizerMixin.get_feature_names_out
except ImportError:
    pass

from evaluation.data import DataLoader
from evaluation.evaluation import Trainer
from evaluation.results import Results
