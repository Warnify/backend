import re
def extract_urls(text):
    # Regular expression pattern to match URLs
    url_pattern = r'https?://\S+|www\.\S+'
    # Find all URLs in the text using the pattern
    urls = re.findall(url_pattern, text)
    # Remove URLs from the text
    cleaned_text = re.sub(url_pattern, '', text)
    # Return a dictionary containing the cleaned text and extracted URLs
    return {"text": cleaned_text.strip(), "urls": urls}
