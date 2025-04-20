import regex as re

def contains_any_word(target_string, word_list):
    # Construct a pattern that matches any word from the word_list as a whole word.
    pattern = r'\b(?:' + '|'.join(map(re.escape, word_list)) + r')\b'
    
    # Compile the regex; adding re.IGNORECASE if case-insensitive search is desired.
    regex = re.compile(pattern, re.IGNORECASE)
    
    # Search the target_string for any match; returns a match object if found, None otherwise.
    match = regex.search(target_string)
    
    return match is not None

# Example usage:
words = ["apple", "banana", "cherry"]
test_string = "I had an Apple pie yesterday!"

if contains_any_word(test_string, words):
    print("A word from the list was found in the string.")
else:
    print("No words from the list were found in the string.")