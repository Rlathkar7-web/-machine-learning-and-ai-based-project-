import string
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
import warnings
warnings.filterwarnings("ignore")

# 1. LOAD AND PREPARE DATA
with open("data/full_set.txt") as f:
    content = f.readlines()

# Remove leading/trailing whitespace
content = [x.strip() for x in content]

# Separate sentences and labels
sentences = [x.split("\t")[0] for x in content]
labels = [x.split("\t")[1] for x in content]

# Transform labels: 0/1 → -1/+1 (standard for signed labels)
y = np.array(labels, dtype='int8')
y = 2 * y - 1

print(f"Total sentences loaded: {len(content)}")

# 2. TEXT PREPROCESSING
def full_remove(x, removal_list):
    """Replace every character in removal_list with a space."""
    for w in removal_list:
        x = x.replace(w, ' ')
    return x

# Remove digits
digits = [str(x) for x in range(10)]
digit_less = [full_remove(x, digits) for x in sentences]

# Remove punctuation
punc_less = [full_remove(x, list(string.punctuation)) for x in digit_less]

# Lowercase everything
sents_lower = [x.lower() for x in punc_less]

# Small custom stop-word set (kept exactly as in original project)
stop_set = {'the', 'a', 'an', 'i', 'he', 'she', 'they', 'to', 'of', 'it', 'from', 'one', 'two', 'ft'}

# Remove stop words
sents_split = [x.split() for x in sents_lower]
sents_processed = [" ".join(list(filter(lambda a: a not in stop_set, x))) for x in sents_split]

print("Final preprocessed sentences (first 10):", sents_processed[:10])

#3. BAG-OF-WORDS FEATURES
vectorizer = CountVectorizer(analyzer="word",
                             tokenizer=None,
                             preprocessor=None,
                             stop_words=None,
                             max_features=4500)

data_features = vectorizer.fit_transform(sents_processed)
data_mat = data_features.toarray()   # dense matrix (small dataset, fine)

# Note: We do NOT manually append a column of 1s for bias.
# SGDClassifier automatically learns the intercept term.

# 4. TRAIN / TEST SPLIT (balanced test set)
np.random.seed(0)
test_inds = np.append(
    np.random.choice((np.where(y == -1))[0], 250, replace=False),
    np.random.choice((np.where(y == 1))[0], 250, replace=False)
)
train_inds = list(set(range(len(labels))) - set(test_inds))

train_data = data_mat[train_inds, :]
train_labels = y[train_inds]

test_data = data_mat[test_inds, :]
test_labels = y[test_inds]

print("Train shape:", train_data.shape)
print("Test shape:", test_data.shape)

#5. FIT LOGISTIC REGRESSION
clf = SGDClassifier(loss="log", penalty="none")
clf.fit(train_data, train_labels)

# Extract weights and bias (bias is handled internally)
w = clf.coef_[0, :]
b = clf.intercept_

# Predictions and error rates
preds_train = clf.predict(train_data)
preds_test = clf.predict(test_data)

errs_train = np.sum((preds_train > 0.0) != (train_labels > 0.0))
errs_test = np.sum((preds_test > 0.0) != (test_labels > 0.0))

print("Training error:", float(errs_train) / len(train_labels))
print("Test error:", float(errs_test) / len(test_labels))

# 6. MARGIN ANALYSIS
def margin_counts(clf, test_data, gamma):
    """Fraction of test points whose predicted probability is at least gamma away from 0.5."""
    preds = clf.predict_proba(test_data)[:, 1]
    margin_inds = np.where((preds > (0.5 + gamma)) | (preds < (0.5 - gamma)))[0]
    return float(len(margin_inds))

def margin_errors(clf, test_data, test_labels, gamma):
    """Error rate only on points that have margin >= gamma."""
    preds = clf.predict_proba(test_data)[:, 1]
    margin_inds = np.where((preds > (0.5 + gamma)) | (preds < (0.5 - gamma)))[0]
    num_errors = np.sum((preds[margin_inds] > 0.5) != (test_labels[margin_inds] > 0.0))
    return float(num_errors) / len(margin_inds) if len(margin_inds) > 0 else 0.0


# Plot 1: Coverage vs margin
gammas = np.arange(0, 0.5, 0.01)
f = np.vectorize(lambda g: margin_counts(clf, test_data, g))
plt.plot(gammas, f(gammas) / 500.0, linewidth=2, color='green')
plt.xlabel('Margin', fontsize=14)
plt.ylabel('Fraction of points above margin', fontsize=14)
plt.title('Coverage vs Margin')
plt.grid(True)
plt.savefig('margin_coverage.png')
plt.show()

# Plot 2: Error rate vs margin
f = np.vectorize(lambda g: margin_errors(clf, test_data, test_labels, g))
plt.plot(gammas, f(gammas), linewidth=2)
plt.ylabel('Error rate', fontsize=14)
plt.xlabel('Margin', fontsize=14)
plt.title('Error Rate vs Margin (high-margin points)')
plt.grid(True)
plt.savefig('margin_error_rate.png')
plt.show()


