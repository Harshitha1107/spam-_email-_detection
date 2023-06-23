import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download stopwords if not already downloaded
nltk.download('stopwords')

def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Join tokens back into a single string
    processed_text = ' '.join(tokens)
    
    return processed_text

# Sample email data
spam_emails = [
    "Get rich quick!",
    "Hi, I am a Nigerian prince and I have a business proposal for you.",
    "You have won a free vacation. Claim now!",
]

ham_emails = [
    "Hello, can we meet tomorrow to discuss the project?",
    "Reminder: Team meeting at 2 PM today.",
    "Please find attached the report for your reference.",
]

# Prepare the training data
all_emails = spam_emails + ham_emails
labels = ['spam'] * len(spam_emails) + ['ham'] * len(ham_emails)

processed_emails = [preprocess_text(email) for email in all_emails]

# Create a feature vector
vectorizer = CountVectorizer()
feature_vector = vectorizer.fit_transform(processed_emails)

# Train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(feature_vector, labels)

# Test the classifier
test_email = "Congratulations! You have won a million dollars!"
processed_test_email = preprocess_text(test_email)
test_vector = vectorizer.transform([processed_test_email])
prediction = classifier.predict(test_vector)[0]

# Output the result
if prediction == 'spam':
    print("The email is classified as spam.")
else:
    print("The email is classified as ham.")
